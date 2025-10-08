import os
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status, serializers

from . import utils
from .logic import extract_items_from_image


SERVER_URL = os.getenv("SERVER_URL")


class ExtractItemsSerializer(serializers.Serializer):
    file = serializers.ImageField()

    class Meta:
        fields = ("file",)


class ExtractItemsView(APIView):
    parser_classes = APIView.parser_classes  # keeps default; DRF handles multipart

    def post(self, request):
        serializer = ExtractItemsSerializer(data=request.data)
        if serializer.is_valid():
            file = serializer.validated_data.pop("file")

            resp = extract_items_from_image(file)

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
            return Response(data=items, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
