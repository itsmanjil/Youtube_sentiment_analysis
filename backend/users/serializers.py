from django.contrib.auth.password_validation import validate_password
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

    def validate_password(self, value):
        # Enforce settings.AUTH_PASSWORD_VALIDATORS (min length, common
        # passwords, etc.) — DRF's ModelSerializer does not run these
        # automatically for a plain CharField.
        validate_password(value)
        return value

    def validate(self, attrs):
        if attrs.get('password') != attrs.get('password2'):
            raise serializers.ValidationError({'password': 'Password must match!'})
        return attrs

    def save(self):
        user = NewUser(
            email = self.validated_data['email'],
            user_name = self.validated_data['user_name'],
            is_registered = True,
        )

        password = self.validated_data['password']
        user.set_password(password)
        user.save()
        return user
