import io
from PIL import Image


def byte_to_pillow(image_bytes):
    raw = image_bytes.read()
    image = Image.open(io.BytesIO(raw))
    return raw, image, image.width, image.height


def pillow_to_byte(image):
    out = io.BytesIO()
    image.save(out, format="JPEG", optimize=True, quality=90)
    out.seek(0)
    out = out.getvalue()
    return out
