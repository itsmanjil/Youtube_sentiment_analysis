from copyreg import constructor
from email import message
from turtle import update
from django.core.exceptions import ValidationError
from django.contrib.auth.hashers import check_password
from django.contrib.auth import login, logout
from rest_framework import serializers
from rest_framework.authtoken.models import Token
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.authtoken.models import Token
from rest_framework.permissions import AllowAny, IsAuthenticated
from django.contrib.auth.decorators import login_required
# from users.authentication import expires_in

from users.models import NewUser
from app.models import YouTubeAnalysis
from .serializers import RegistrationSerializer
# from .authentication import token_expire_handler, expires_in
from .authentication import ExpiringTokenAuthentication

import pytz
import datetime
import logging

logger = logging.getLogger(__name__)




from django.db import IntegrityError
@api_view(["POST"])
# @permission_classes([AllowAny])
def registration_view(request):
    data = {}
    serializer = RegistrationSerializer(data=request.data)
    logger.debug("registration_view serializer=%s data=%s", serializers, data)
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
def login_view(request):
    data = {}
    email = request.data['email']
    password = request.data['password']

    logger.debug("login_view email=%s", email)

    try:
        User = NewUser.objects.get(email=email)
        logger.debug("login_view user id=%s", User.id)
    except BaseException as e:
        raise serializers.ValidationError({"400":f'{str(e)}'})
    
    utc_now = datetime.datetime.utcnow()
    utc_now = utc_now.replace(tzinfo=pytz.utc)



    if not check_password(password, User.password):
        raise serializers.ValidationError({"error": "Incorrect login credentials"})

    if User:
        if User.is_active:
            login(request, User)
            data["message"] = "User logged in."
            data["email"] = User.email
            data["id"] = User.id
            data["is_registered"] = User.is_registered
            res = {"data": data, 
            }

            return Response(res)

        else:
            raise ValidationError({"message":"User not active"})
    else:
        raise ValidationError({"message":"Account doesnot exists."})


@api_view(["GET",])
@permission_classes([IsAuthenticated])
def logout_view(request):
#    request.user.auth_token.delete()
   logout(request)
   return Response({"message":"User logged out"})

@api_view(["GET",])
def get_user(request, id):
    user = NewUser.objects.get(id=id)
    logger.debug("get_user request_user=%s target_user=%s", request.user, user)

    youtubeData = YouTubeAnalysis.objects.filter(user=user).order_by('-id')
    searched_lists = []
    for analysis in youtubeData:
        searched_lists.append({
            'video_id': analysis.video.video_id,
            'title': analysis.video.title
        })

    isSameUser = request.user == user
    if user.is_authenticated & isSameUser:
        if user.id == id:
            return Response({
            "user_name":user.user_name,
            "email":user.email,
            "id":user.id,
            "searched_list": searched_lists
            #    "expires_in": expires_in(request.auth)
            })
    else:
        return Response({
            "message":"You can only get your information"
        })
