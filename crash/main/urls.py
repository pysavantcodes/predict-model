from django.urls import path
from .views import CrashAPIView

urlpatterns = [
    path('api/crash', CrashAPIView.as_view(), name='crash'),
]
