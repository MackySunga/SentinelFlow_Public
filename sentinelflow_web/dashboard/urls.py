from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_dataset, name='upload_dataset'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('models/', views.model_status, name='model_status'),
    path('reports/', views.reports, name='reports'),
]
