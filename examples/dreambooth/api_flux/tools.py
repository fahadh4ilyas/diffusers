import io, base64
from PIL import Image

def pil_image_to_bytes(image: Image.Image, format: str = 'PNG') -> bytes:
    """
    Converts a PIL Image object to a bytes array.

    Args:
        image: The PIL Image object to convert.
        format: The image format for saving (e.g., 'PNG', 'JPEG').

    Returns:
        A bytes object representing the image.
    """
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()

def base64_image_to_pil_image(base64_img: str) -> Image.Image:
    """
    Converts a base64 encoded string to a PIL Image object.

    Args:
        base64_img: The base64 encoded string with possibly metadata at the beginning of string.

    Returns:
        A PIL Image object.
    """
    header, encoded = base64_img.split(",", 1) if "," in base64_img else ("", base64_img)
    img_data = base64.b64decode(encoded)
    return Image.open(io.BytesIO(img_data))


    