from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken, TokenError

from core.auth_cookies import REFRESH_COOKIE_NAME, clear_refresh_cookie
from users.models import NewUser
from app.models import YouTubeAnalysis
from .serializers import RegistrationSerializer
import logging

logger = logging.getLogger(__name__)
@api_view(["POST"])
@permission_classes([AllowAny])
def registration_view(request):
    data = {}
    serializer = RegistrationSerializer(data=request.data)
    if serializer.is_valid():
        account = serializer.save()
        account.is_active = True
        account.save()
        data["email"] = account.email
        data["user_name"] = account.user_name
        # data["token"] = token
        return Response({'message':'User registered', 'data':data},status=status.HTTP_200_OK)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
 

@api_view(["POST",])
@permission_classes([AllowAny])
def logout_view(request):
   # The refresh token lives in an httpOnly cookie (core/auth_cookies.py),
   # not the request body, so the browser attaches it automatically.
   refresh = request.COOKIES.get(REFRESH_COOKIE_NAME)
   if not refresh:
       return Response(
           {"message": "refresh token is required"},
           status=status.HTTP_400_BAD_REQUEST,
       )

   try:
       token = RefreshToken(refresh)
       token.blacklist()
   except TokenError:
       return Response(
           {"message": "Invalid refresh token"},
           status=status.HTTP_400_BAD_REQUEST,
       )

   response = Response({"message": "User logged out"})
   clear_refresh_cookie(response)
   return response

@api_view(["GET",])
@permission_classes([IsAuthenticated])
def get_user(request, id):
    if request.user.id != id:
        return Response(
            {"message":"You can only get your information"},
            status=status.HTTP_403_FORBIDDEN,
        )

    user = get_object_or_404(NewUser, id=id)
    logger.debug("get_user request_user=%s target_user=%s", request.user, user)
    # select_related('video') avoids one extra query per analysis (each row
    # reads analysis.video.*), and the [:20] bound keeps the payload/query
    # cost flat regardless of history size. The frontend (Profile.jsx,
    # Dashboard.jsx) only renders this as a fixed scroll list, so a heavy
    # user's full history was never needed here. Matches the sibling
    # endpoint app/views.py::get_user_youtube_analyses.
    youtubeData = (
        YouTubeAnalysis.objects
        .filter(user=user)
        .select_related('video')
        .order_by('-id')[:20]
    )
    searched_lists = []
    for analysis in youtubeData:
        searched_lists.append({
            'video_id': analysis.video.video_id,
            'title': analysis.video.title
        })

    return Response({
        "user_name":user.user_name,
        "email":user.email,
        "id":user.id,
        "searched_list": searched_lists
        #    "expires_in": expires_in(request.auth)
    })
