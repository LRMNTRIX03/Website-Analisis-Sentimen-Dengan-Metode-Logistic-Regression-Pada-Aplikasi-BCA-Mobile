from django.contrib import admin
from django.urls import path
from . import views


urlpatterns = [
    path('inputdata/', views.inputdata, name='sentimen.inputdata'),
    
    # Route untuk halaman stopword
    path('stopword/', views.stopword, name='sentimen.stopword'),
    path('stopword/create/', views.stopword_create, name='sentimen.stopword.create'),
    path('stopword/<int:id>/edit/', views.stopword_edit, name='sentimen.stopword.edit'),
    path('stopword/<int:id>/delete/', views.stopword_delete, name='sentimen.stopword.delete'),
 

    # Route untuk halaman Slangword
    path('slangword/', views.slangword, name='sentimen.slangword'),
    path('slangword/create/', views.slangword_create, name='sentimen.slangword.create'),
    path('slangword/<int:id>/edit/', views.slangword_edit, name='sentimen.slangword.edit'),
    path('slangword/<int:id>/delete/', views.slangword_delete, name='sentimen.slangword.delete'),

    # Route untuk halaman Preprocessing
    path('preprocessing/', views.preprocessing, name='sentimen.preprocessing'),
    path('labelisasi/', views.labelisasi, name='sentimen.labelisasi'),
    path('tfidf/', views.tfidf, name='sentimen.tfidf'),
    path('klasifikasi/', views.klasifikasi, name='sentimen.klasifikasi'),
    path('evaluasi/', views.evaluasi, name='sentimen.evaluasi'),
    path('kalimat/', views.kalimat, name='sentimen.kalimat'),
]