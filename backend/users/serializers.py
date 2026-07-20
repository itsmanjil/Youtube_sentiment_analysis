from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError as DjangoValidationError
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

    def validate(self, attrs):
        if attrs.get('password') != attrs.get('password2'):
            raise serializers.ValidationError({'password': 'Password must match!'})

        # Enforce settings.AUTH_PASSWORD_VALIDATORS (min length, common
        # passwords, etc.) — DRF's ModelSerializer does not run these
        # automatically for a plain CharField. This must happen here, not in
        # a per-field validate_password(), because UserAttributeSimilarityValidator
        # needs the account's own attributes to compare against, and a
        # field-level validator runs before the other fields are in scope. A
        # transient, unsaved NewUser carries email/user_name so a password
        # equal (or very similar) to the email is rejected instead of silently
        # passing, as it did when validate_password() was called with no user.
        candidate = NewUser(
            email=attrs.get('email', ''),
            user_name=attrs.get('user_name', ''),
        )
        try:
            validate_password(attrs.get('password'), user=candidate)
        except DjangoValidationError as exc:
            raise serializers.ValidationError({'password': list(exc.messages)})

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
