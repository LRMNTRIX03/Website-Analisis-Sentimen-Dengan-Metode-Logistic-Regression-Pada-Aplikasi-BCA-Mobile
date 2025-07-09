from django.contrib import admin
from django.urls import path
from django.conf.urls import include
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('authapp.urls')),
    path('dashboard/', include('sentimenapp.urls')),

]
