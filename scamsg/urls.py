from django.urls import path
from . import views

urlpatterns = [
    path('detect_scam_gnb/', views.detect_scam_using_gnb, name='detect_scam_using_gnb'),
    path('detect_scam_lstm/', views.detect_scam_lstm, name='detect_scam_lstm'),
]