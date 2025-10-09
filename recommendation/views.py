from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, serializers

from .logic import get_llm_response

class RecommendationView(APIView):
    def post(self, request):
        print(vars(request))
        print(request.data)
        query = request.data.pop("query")
        occasions = request.data.pop("occasions")
        items = request.data.pop("selected_item_captions")
        catalog = request.data.pop("other_item_captions")
        
        print(f"{query=}")
        print(f"{occasions=}")
        print(f"{items=}")
        print(f"{catalog=}")
        
        recom = get_llm_response(query, occasions, items, catalog)
        print(recom)
        
        return Response(recom, status=status.HTTP_200_OK)
        