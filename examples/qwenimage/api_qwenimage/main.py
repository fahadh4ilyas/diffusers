# fastapi_stream_wav.py
import os
import time
import asyncio
import threading
import traceback
import timeit
from typing import Optional, Any, List, Dict, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request, Body, HTTPException
from fastapi.responses import ORJSONResponse, Response, RedirectResponse

import torch
import numpy as np
from contextlib import asynccontextmanager

from .config import config, LOGGER_ACCESS, LOGGER
from .tools import pil_image_to_bytes, base64_image_to_pil_image

from diffusers import QwenImagePipeline, QwenImageImg2ImgPipeline

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

class CommunicationQueue:

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.ended = False

    async def put(self, item: bytes):
        await self.queue.put(item)

    def put(self, item: bytes):
        # sync put
        self.queue.put_nowait(item)
    
    async def get(self) -> bytes:
        return await self.queue.get()

    async def end(self):
        self.ended = True

class DataQueue:

    def __init__(self, max_batch_size: int, model: Union[QwenImagePipeline, QwenImageImg2ImgPipeline], force_lora: bool, lora_path: Optional[str] = None, lora_keywords: Optional[str] = None, include_keywords: bool = False):
        self.active_queue: List[Dict[str, Any]] = []
        self.queue: List[Dict[str, Any]] = []
        self.max_batch_size = max_batch_size
        self.model = model
        self.force_lora = force_lora
        self.lora_path = lora_path
        self.lora_keywords = sorted(lora_keywords.split(',') if lora_keywords else [], key=len, reverse=True)
        self.include_keywords = include_keywords
        self.image_input = model.__class__ == QwenImageImg2ImgPipeline
        if force_lora:
            self.model.load_lora_weights(self.lora_path)
        self.vae_scale = self.model.vae_scale_factor
        self.default_sample_size = self.model.default_sample_size
        self.stopped = False
        self.loop = asyncio.get_event_loop()
    
    def put(
        self,
        text: str,
        negative_text: Optional[str],
        images: Optional[List[str]],
        strength: Optional[float],
        num_inference_steps: int,
        height: Optional[int],
        width: Optional[int],
        queue: CommunicationQueue
    ):
        global lock
        lock.acquire()
        use_lora = False
        if not self.force_lora and self.lora_keywords:
            for kw in self.lora_keywords:
                if text.startswith(kw):
                    use_lora = True
                    if not self.include_keywords:
                        text = text.removeprefix(kw).lstrip()
                    break
        LOGGER.debug(f"DataQueue put: use_lora={use_lora}, text='{text}'")
        with torch.no_grad():
            prompt_embeds, prompt_embeds_mask = self.model.encode_prompt(prompt=text)
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None
            true_cfg_scale = 1.0
            if negative_text:
                negative_prompt_embeds, negative_prompt_embeds_mask = self.model.encode_prompt(prompt=negative_text)
                true_cfg_scale = 4.0
            if images:
                images = [base64_image_to_pil_image(img) for img in images]
                init_image = self.model.preprocess_image(images, height, width)
        lock.release()
        height = height or self.default_sample_size * self.vae_scale
        width = width or self.default_sample_size * self.vae_scale
        item = {
            "queue": queue,
            "use_lora": use_lora and self.lora_path is not None,
            "sigmas": np.linspace(1.0, 1 / num_inference_steps, num_inference_steps),
            "step": 0,
            "finished": False,
        }
        item['next_inputs'] = {
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
            "true_cfg_scale": true_cfg_scale,
            "height": height,
            "width": width,
            "latents": None,
            "sigmas": item["sigmas"][item["step"]:],
            "output_type": "latent",
            "max_inference_steps": 1,
            "verbose": False
        }
        if self.image_input:
            if strength is None:
                item['next_inputs']['constant_t_start'] = self.model.calculate_t_start(num_inference_steps=num_inference_steps)
            else:
                item['next_inputs']['constant_t_start'] = self.model.calculate_t_start(num_inference_steps=num_inference_steps, strength=strength)
            if images:
                item['next_inputs']['init_image'] = init_image
            if strength is not None:
                item['next_inputs']['strength'] = strength
        if len(self.active_queue) < self.max_batch_size:
            self.active_queue.append(item)
        else:
            self.queue.append(item)
    
    def check_queue(self) -> bool:
        "remove finished items from active queue, fill from waiting queue, but remove cancelled items first from queue"
        removed_count = 0
        for i in range(len(self.active_queue)-1, -1, -1):
            if self.active_queue[i]['finished']:
                self.active_queue.pop(i)
                removed_count += 1
        for i in range(len(self.queue)-1, -1, -1):
            if self.queue[i]['finished']:
                self.queue.pop(i)
        for _ in range(removed_count):
            if self.queue:
                item = self.queue.pop(0)
                self.active_queue.append(item)
        return len(self.active_queue) > 0
    
    @property
    def is_empty(self):
        return len(self.active_queue) == 0
    
    def log_status(self):
        LOGGER.info(f"DataQueue status: active={len(self.active_queue)}, waiting={len(self.queue)}")

    def replace_item(self, index: int, item: Dict[str, Any]):
        if 0 <= index < len(self.active_queue):
            self.active_queue[index] = item

    def set_stopped(self, stopped: bool):
        self.stopped = stopped

    def single_step(self, item: Dict[str, Any]):
        "process a single item for one step"
        
        if item['finished']:
            return item
        queue: CommunicationQueue = item['queue']
        if queue.ended:
            return {
                "finished": True,
                "next_inputs": None
            }

        next_inputs = item['next_inputs']
        use_lora = item['use_lora']
        if use_lora:
            LOGGER.debug("Loading LoRA weights")
            self.model.load_lora_weights(self.lora_path)
        output = self.model(**next_inputs).images
        if use_lora:
            LOGGER.debug("Unloading LoRA weights")
            self.model.unload_lora_weights()
        if next_inputs["output_type"] == "latent":
            latents = output
            next_inputs["latents"] = latents
            item["step"] += 1
            next_inputs["sigmas"] = item["sigmas"][item["step"]:]
            if len(next_inputs["sigmas"]) == 1 or ("constant_t_start" in next_inputs and len(next_inputs["sigmas"]) - next_inputs["constant_t_start"] == 1):
                next_inputs["output_type"] = "pil"
            if queue.ended:
                return {
                    "finished": True,
                    "next_inputs": None
                }
            return item
        image_bytes = pil_image_to_bytes(output[0])
        if queue.ended:
            return {
                "finished": True,
                "next_inputs": None
            }
        self.loop.call_soon_threadsafe(queue.put, image_bytes)
        return {
            "finished": True,
            "next_inputs": None
        }
    
    def infinite_loop_step(self):
        "infinite loop in another thread and the exit if interrupted or get killed"

        log_counter = 0
        try:
            while True:
                if self.stopped:
                    break
                now = time.time()
                if log_counter >= 15.0:
                    self.log_status()
                    log_counter = 0
                if self.is_empty:
                    time.sleep(0.1)
                    log_counter += (time.time() - now)
                    continue
                self.check_queue()
                for i in range(len(self.active_queue)):
                    item = self.active_queue[i]
                    try:
                        step_output = self.single_step(item)
                    except:
                        exception = traceback.format_exc()
                        LOGGER.error(f"DataQueue infinite_loop_step single_step exception: {exception}")
                        item['queue'].put(f'Error: {exception}'.encode('utf-8'))
                        step_output = {
                            "finished": True,
                            "next_inputs": None
                        }
                    self.replace_item(i, step_output)
                torch.cuda.empty_cache()
                log_counter += (time.time() - now)
        except KeyboardInterrupt:
            LOGGER.info("infinite_loop_step interrupted, exiting")
        except Exception as exc:
            LOGGER.error(f"infinite_loop_step exception: {traceback.format_exc()}")

executor = ThreadPoolExecutor(max_workers=4)

# --- model placeholders (load your model in startup) ---
model: Union[QwenImagePipeline, QwenImageImg2ImgPipeline, None] = None
data_queue: Optional[DataQueue] = None
lock = threading.Lock()
voice_list = {p.stem.split('_')[0]: p for p in Path(os.path.join(ROOT_DIR, 'sample-voices')).glob('*.wav')}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, data_queue, executor

    if config.image_input:
        model = QwenImageImg2ImgPipeline.from_pretrained(config.model_path, torch_dtype=torch.bfloat16).to('cuda')
    else:
        model = QwenImagePipeline.from_pretrained(config.model_path, torch_dtype=torch.bfloat16).to('cuda')
    data_queue = DataQueue(
        max_batch_size=config.max_batch_size,
        model=model,
        force_lora=config.force_lora,
        lora_path=config.lora_path,
        lora_keywords=config.lora_keywords,
        include_keywords=config.include_keywords,
    )
    loop = asyncio.get_event_loop()
    infinite_thread = loop.run_in_executor(executor, data_queue.infinite_loop_step)
    LOGGER.info("Startup: model should be loaded here")
    yield
    data_queue.set_stopped(True)
    await infinite_thread
    LOGGER.info("Shutdown: clean up resources if needed")

app = FastAPI(title='QwenImage API',
    description='API for generating images using QwenImage model.',
    version='1.0.0',
    lifespan=lifespan)


@app.exception_handler(Exception)
async def value_error_handler(request: Request, exc: Exception):
    return ORJSONResponse({
        'error': str(exc),
        'traceback': "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        'status_code': 500
    }, status_code=500)


@app.middleware("http")
async def logging_request(request: Request, call_next):

    client_data = ''
    if request.client:
        client_data = f'{request.client.host}:{request.client.port}'
    LOGGER_ACCESS.info(f'{client_data} - "{request.method.upper()} {request.url.path} {request.url.scheme.upper()}/1.1" START')
    params = str(request.query_params)
    body = await request.body()
    if params:
        LOGGER_ACCESS.info(f'{client_data} - "{request.method.upper()} {request.url.path} {request.url.scheme.upper()}/1.1" PARAMS: {params}')
    if body:
        LOGGER_ACCESS.info(f'{client_data} - "{request.method.upper()} {request.url.path} {request.url.scheme.upper()}/1.1" BODY: {(await request.body())[:256]}')

    start = timeit.default_timer()
    request.state.is_disconnected = request.is_disconnected
    response: Response = await call_next(request)
    response.headers["X-Process-Time"] = f'{timeit.default_timer() - start:.6f}'

    return response


@app.post("/generate")
async def generate_image(
    request: Request,
    text: str = Body(..., examples=["Create a clean, modern, Gen-Z styled digital safety infographic in Indonesian for 'Aware Daily'. Instagram square format (1080x1080). Color palette: Background #F4F4F0 (soft neutral), Primary Blue #2A4A85 (digital navy), Accent Blue #6C8BC7 (soft bright blue), Highlight Mint #A7E2C5 (mint green), Text Color #1C1C1C, Icon Outline #2A4A85. \n\nText content (EXACT COPY):\nTitle: 'KADANG, YANG KAMU LIHAT DI INTERNET TIDAK SE-BENAR ITU.'\nSubtitle: 'Ada hal-hal yang sengaja dibuat untuk mempengaruhi cara kamu berpikir.'\nBullet points:\n\u2022 Konten yang memicu emosi supaya kamu cepat bereaksi\n\u2022 Informasi yang membuat kamu merasa harus berpihak\n\u2022 Postingan yang sengaja dibikin terlihat \"darurat\"\n\u2022 Timeline yang hanya menunjukkan satu sudut pandang\nHighlight box: 'Latih diri untuk berhenti sejenak dan cek ulang informasi.'\nFooter: '@awaredaily'\n\nDesign style: Modern, digital, calming aesthetic with rounded shapes, soft gradients, minimal but stylish. Clean Gen Z layout with soft outline icons next to each bullet point. No human faces. Credible yet friendly appearance. Use the exact colors specified: #F4F4F0 background, #2A4A85 primary blue, #6C8BC7 accent blue, #A7E2C5 mint green, #1C1C1C text color."]),
    negative_text: Optional[str] = Body(None, examples=[None]),
    images: Optional[List[str]] = Body(None, examples=[None]),
    strength: Optional[float] = Body(None, examples=[None]),
    num_inference_steps: int = Body(50),
    height: Optional[int] = Body(None, examples=[None]),
    width: Optional[int] = Body(None, examples=[None]),
):
    """
    Generate an image based on the provided text prompt and optional images.
    Args:
        text: The text prompt for image generation.
        negative_text: The negative text prompt to avoid certain features in the generated image.
        images: Optional list of base64 encoded images to condition the generation.
        strength: Strength of the image conditioning (only for img2img).
        num_inference_steps: Number of inference steps for generation.
        height: Height of the generated image.
        width: Width of the generated image.
    """
    global data_queue

    output_queue = CommunicationQueue()

    data_queue.put(text, negative_text, images, strength, num_inference_steps, height, width, output_queue)

    async def disconnect_watcher():
        try:
            is_disc = await request.state.is_disconnected()
        except Exception:
            # treat errors as disconnected
            is_disc = True
        if is_disc:
            await output_queue.end()

    disconnect_task = asyncio.create_task(disconnect_watcher())

    image_bytes = await output_queue.get()

    if image_bytes.startswith(b'Error:'):
        raise HTTPException(status_code=500, detail=image_bytes.decode('utf-8').removeprefix('Error: '))

    return Response(content=image_bytes, media_type='image/png')

@app.get('/', include_in_schema=False)
async def redirect():

    return RedirectResponse(app.root_path+'/docs')
