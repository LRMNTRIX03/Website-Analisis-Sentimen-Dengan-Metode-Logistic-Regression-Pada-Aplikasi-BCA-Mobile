from django.contrib import admin
from django.urls import path, include
from . import views


urlpatterns = [
    path('', views.login_view, name='auth.login' ),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard_view, name='sentimen.index'), 
    path('', include('sentimenapp.urls')),
]
