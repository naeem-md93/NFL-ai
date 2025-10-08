import os
import json
import base64
import requests
from PIL import Image
from io import BytesIO

from . import utils


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL_NAME = os.getenv("OPENROUTER_MODEL_NAME")

SYSTEM_PROMPT = """
SYSTEM:
You are an accurate, conservative computer-vision + fashion-attribute extractor.
Given an image (provided as a visual input) and an optional short caption describing the image, your job is to detect garment items of type "shirt" and "pants" only, describe each detected garment, and return the exact JSON array described below. Be precise with bounding boxes and brief in descriptions.

Important rules:
- Output **only** valid JSON (no surrounding text, no backticks, no explanations).
- The JSON must be a top-level array of objects. Each object must contain exactly these keys: "type", "description", "bbox".
- "type" must be either the string "shirt" or "pants".
- "description" must be a concise factual textual description (6–18 words) of the garment, mentioning visible attributes such as color, pattern, sleeve length or pant cut when discernible.
- "bbox" must be normalized coordinates in [0.0, 1.0]: `[x, y, w, h]` where:
  - x,y are the top-left corner relative to the original image (0.0 = left or top, 1.0 = right or bottom),
  - w and h are width and height as fractions of the original image width/height.
  - All numbers must be floats formatted with **three decimal places** (for example `0.123`).
- Coordinates must follow image convention: origin (0,0) at top-left, x to right, y down.
- Compute normalized values relative to the original image size. If `image_width` and `image_height` metadata are provided, use those dimensions; otherwise use the visual image dimensions available to you.
- Do not invent garments not visually present. If a detected garment is ambiguous (e.g., partially occluded), include it only if you are reasonably certain. Do not output confidence fields — only items you are confident about.
- If no shirts or pants are detected, output an empty array: `[]`.
- **Do not return duplicate garments.** Avoid returning multiple boxes for the same visible garment.
- Keep each "description" factual and avoid subjective claims (e.g., don't guess brand or exact material unless visually obvious).

You will receive:
1) the image (attached as the visual input to the model)
2) a short textual caption describing the image (may be omitted)
3) optionally: image metadata in JSON with keys: image_width (int), image_height (int)

Task:
Detect garment items of type "shirt" or "pants" visible in the image. For each detected garment return an object with:
- "type": "shirt" or "pants"
- "description": a concise factual description (6–18 words) including visible color, pattern, sleeve length or pant cut when discernible
- "bbox": normalized `[x, y, w, h]` as described above (floats with 3 decimal places)

INPUT:
Caption: "{{caption_text}}" or ""
(Optional) Image metadata JSON: {{ "image_width": IMAGE_WIDTH, "image_height": IMAGE_HEIGHT }}

OUTPUT:
Return only a JSON array like:

[
  {
    "type": "shirt",
    "description": "white short-sleeve cotton t-shirt with small logo",
    "bbox": [0.221, 0.120, 0.354, 0.270]
  },
  {
    "type": "pants",
    "description": "dark blue slim-fit jeans, ankle length",
    "bbox": [0.180, 0.520, 0.420, 0.380]
  }
]

Notes:
- Every bbox must be normalized (0.000 - 1.000) and given as [x, y, w, h] with three decimal digits.
- Do not include any extra keys, metadata, or text — only the specified array of objects.
- If multiple detected boxes overlap heavily (near-duplicates), merge them and return a single representative detection.
- If you must assume image dimensions because metadata is absent, compute normalization relative to the image you were given and state nothing — just output the normalized bboxes as required.
"""

USER_PROMPT = """
USER:
Caption: {caption_text}
(Optional) Image metadata JSON: {image_metadata}
"""


def get_response(system_prompt, user_text, user_image):

    try:
        resp = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
            "model": OPENROUTER_MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": user_image}}
                ]},
            ],
            "temperature": 0,
            })
        )

        resp = resp.json()
        resp = resp["choices"][0]["message"]["content"]
        resp = resp.replace("```json", "").replace("```", "")
        resp = json.loads(resp)
        print(resp)
    except Exception as e:
        print(resp)
        print(repr(e))
        resp = []
    return resp


def extract_items_from_image(file_obj):

    (width, height) = file_obj.image.size
    data_uri = utils.get_image_data_uri(file_obj)
    
    result = get_response(
        SYSTEM_PROMPT,
        USER_PROMPT.format(
            caption_text="",
            image_metadata={"image_width": width, "image_height": height}
        ),
        data_uri
    )

    return result