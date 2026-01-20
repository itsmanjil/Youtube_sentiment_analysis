from rest_framework import serializers
from .models import NewUser

class RegistrationSerializer(serializers.ModelSerializer):

    password2 = serializers.CharField(write_only=True)

    class Meta:
        model = NewUser
        fields = ['email', 'user_name', 'password', 'password2']
        extra_kwargs = {
            'password' : {'write_only': True}
        }
    
    def save(self):
        user = NewUser(
            email = self.validated_data['email'],
            user_name = self.validated_data['user_name'],
        )
        
        password = self.validated_data['password']
        password2 = self.validated_data['password2']
        if password != password2:
            raise serializers.ValidationError({'password': 'Password must match!'})

        user.set_password(password)
        user.save()
        return user