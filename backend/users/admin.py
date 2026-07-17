from django.contrib import admin
from users.models import NewUser
from django.contrib.auth.admin import UserAdmin
from django.forms import TextInput, Textarea, CharField
from django import forms
from django.db import models
# Register your models here.

class UserAdminConfig(UserAdmin):
    model = NewUser
    search_fields = ('email', 'user_name', 'first_name',)
    list_filter = ('email' ,'user_name', 'first_name', 'last_name', 'is_active', 'is_staff')
    ordering = ('-start_date',)
    list_display = ('email', 'id', 'user_name', 'first_name', 'last_name', 'is_active', 'is_staff', 'is_registered')


admin.site.register(NewUser, UserAdminConfig)

