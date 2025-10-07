import os
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status

from . import utils
from .logic import extract_items_from_image


SEREVR_URL = os.getenv("SEREVR_URL")


class ExtractItemsFromImageView(APIView):
    parser_classes = APIView.parser_classes  # keeps default; DRF handles multipart

    def post(self, request):
        file = request.FILES.get('file')
        print(vars(file))

        if not file:
            return Response({'detail': 'No files provided'}, status=status.HTTP_400_BAD_REQUEST)

        i_raw, i_image, i_width, i_height = utils.byte_to_pillow(file)
        resp = extract_items_from_image(i_raw, i_width, i_height)

        cropped_items = []
        for r in resp:
            cropped_items.append({
                "type": r["type"],
                "caption": r["description"],
                "bbox_x0": r["bbox"][0],
                "bbox_y0": r["bbox"][1],
                "bbox_w": r["bbox"][2],
                "bbox_h": r["bbox"][3],
            })

        return Response(cropped_items, status=status.HTTP_200_OK)
