import io
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