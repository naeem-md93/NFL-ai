from rest_framework.views import APIView


class TryOnView(APIView):
    def post(self, request):
        img = request.FILES.get('image')
        garment = request.data.get('garment')
