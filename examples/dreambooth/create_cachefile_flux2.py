import argparse
import torch
import os
import itertools
import random

from tqdm.auto import tqdm
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from accelerate.logging import get_logger

from diffusers.training_utils import find_nearest_bucket, parse_buckets_string

from transformers import Mistral3ForConditionalGeneration, PixtralProcessor
from diffusers import (
    AutoencoderKLFlux2,
    Flux2Pipeline
)

logger = get_logger(__name__)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Create cache file.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        required=True,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--text_encoder_out_layers",
        type=int,
        nargs="+",
        default=[10, 20, 30],
        help="Text encoder hidden layers to compute the final text embeddings.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where cache file will be saved.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--aspect_ratio_buckets",
        type=str,
        default=None,
        help=(
            "Aspect ratio buckets to use for training. Define as a string of 'h1,w1;h2,w2;...'. "
            "e.g. '1024,1024;768,1360;1360,768;880,1168;1168,880;1248,832;832,1248'"
            "Images will be resized and cropped to fit the nearest bucket. If provided, --resolution is ignored."
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--weight_dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help=(
            "The dtype to use for model weights. If not specified, the weights will be loaded in their original"
            " dtype."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.dataset_name

    return args

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        size=1024,
        center_crop=False,
        buckets=None,
    ):
        self.size = size
        self.center_crop = center_crop

        self.custom_instance_prompts = None

        self.buckets = buckets

        # if --dataset_name is provided or a metadata jsonl file is provided in the local --instance_data directory,
        # we load the training data using load_dataset
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "You are trying to load your data using the datasets library. If you wish to train using custom "
                "captions please install the datasets library: `pip install datasets`."
            )
        # Downloading and loading a dataset from the hub.
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
        # Preprocessing the datasets.
        column_names = dataset["train"].column_names

        # 6. Get the column names for input/target.
        if args.image_column is None:
            image_column = column_names[0]
            logger.info(f"image column defaulting to {image_column}")
        else:
            image_column = args.image_column
            if image_column not in column_names:
                raise ValueError(
                    f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )
        self.instance_images = dataset["train"][image_column]

        if args.caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
        self.custom_instance_prompts = dataset["train"][args.caption_column]

        self.pixel_values = []
        for image in self.instance_images:
            image = exif_transpose(image)
            if not image.mode == "RGB":
                image = image.convert("RGB")

            width, height = image.size

            # Find the closest bucket
            bucket_idx = find_nearest_bucket(height, width, self.buckets)
            target_height, target_width = self.buckets[bucket_idx]
            self.size = (target_height, target_width)

            # based on the bucket assignment, define the transformations
            image = self.train_transform(
                image,
                size=self.size,
                center_crop=args.center_crop,
                random_flip=args.random_flip,
            )
            self.pixel_values.append((image, bucket_idx))

        self.num_instance_images = len(self.instance_images)
        self._length = self.num_instance_images

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image, bucket_idx = self.pixel_values[index % self.num_instance_images]
        example["instance_images"] = instance_image
        example["bucket_idx"] = bucket_idx
        caption = self.custom_instance_prompts[index % self.num_instance_images]
        example["instance_prompt"] = caption

        return example

    def train_transform(self, image, size=(224, 224), center_crop=False, random_flip=False):
        # 1. Resize (deterministic)
        resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        image = resize(image)

        # 2. Crop: either center or SAME random crop
        if center_crop:
            crop = transforms.CenterCrop(size)
            image = crop(image)
        else:
            # get_params returns (i, j, h, w)
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=size)
            image = TF.crop(image, i, j, h, w)

        # 3. Random horizontal flip with the SAME coin flip
        if random_flip:
            do_flip = random.random() < 0.5
            if do_flip:
                image = TF.hflip(image)

        # 4. ToTensor + Normalize (deterministic)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.5], [0.5])
        image = normalize(to_tensor(image))

        return image

def main(args):

    if args.aspect_ratio_buckets is not None:
        buckets = parse_buckets_string(args.aspect_ratio_buckets)
    else:
        buckets = [(args.resolution, args.resolution)]

    train_dataset = DreamBoothDataset(
        size=args.resolution,
        center_crop=args.center_crop,
        buckets=buckets,
    )

    tokenizer = PixtralProcessor.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    weight_dtype = getattr(torch, args.weight_dtype)

    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    vae.requires_grad_(False)
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    text_encoder.requires_grad_(False)

    to_kwargs = {"device": 'cuda'}
    vae.to(**to_kwargs)
    text_encoder.to(**to_kwargs)

    text_encoding_pipeline = Flux2Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=None,
        revision=args.revision,
    )

    def compute_text_embeddings(prompt):
        with torch.no_grad():
            prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
                prompt=prompt,
                max_sequence_length=args.max_sequence_length,
                text_encoder_out_layers=args.text_encoder_out_layers,
            )
        return prompt_embeds.to('cpu'), text_ids.to('cpu')

    def compute_latent_cache(pixel_values):
        with torch.no_grad():
            latents = vae.encode(pixel_values.to('cuda', non_blocking=True, dtype=vae.dtype)).latent_dist
        return latents.mode().to('cpu')

    latent_caches = []

    for i in tqdm(range(len(train_dataset))):
        result = {}
        example = train_dataset[i]
        pixel_values = [example["instance_images"]]
        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        latents = compute_latent_cache(pixel_values)
        result["latents"] = latents
        result["prompt_embeds"] = {}
        for batch_size in range(1, args.train_batch_size + 1):
            prompt = [example["instance_prompt"]] + [''] * (batch_size - 1)
            prompt_embeds, text_ids = compute_text_embeddings(prompt)
            result["text_ids"] = text_ids[:1]
            result["prompt_embeds"][str(batch_size)] = prompt_embeds[:1]
        latent_caches.append(result)

    os.makedirs(args.output_dir, exist_ok=True)

    torch.save(latent_caches, os.path.join(args.output_dir, 'cache.pt'))

if __name__ == "__main__":
    args = parse_args()
    main(args)