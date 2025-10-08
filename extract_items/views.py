import os
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status

from . import utils
from .logic import extract_items_from_image


SERVER_URL = os.getenv("SERVER_URL")


class ExtractItemsView(APIView):
    parser_classes = APIView.parser_classes  # keeps default; DRF handles multipart

    def post(self, request):

        width = int(request.data["width"])
        height = int(request.data["height"])
        mime_type = request.data["mime_type"]
        file = request.FILES["file"]

        resp = extract_items_from_image(file, width, height, mime_type)

        items = []
        for r in resp:
            items.append({
                "type": r["type"],
                "caption": r["description"],
                "box_x": r["bbox"][0],
                "box_y": r["bbox"][1],
                "box_w": r["bbox"][2],
                "box_h": r["bbox"][3],
            })
        print(items)

        return Response(items, status=status.HTTP_200_OK)
