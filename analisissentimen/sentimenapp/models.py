# models.py
from django.db import models

class DataTeks(models.Model):
  
    teks = models.TextField()
   
    

    def __str__(self):
        return self.teks[:50]

class Stopword(models.Model):
    stopwords = models.CharField(max_length=255)

    def __str__(self):
        return self.stopwords
    
class Slangword(models.Model):
    katabaku = models.CharField(max_length=255)
    katatidakbaku = models.CharField(max_length=255)

    def __str__(self):
        return "{self.katabaku} {self.katatidakbaku}"
    
class PreprocessingModel(models.Model):
    teks_awal = models.ForeignKey(DataTeks, on_delete=models.CASCADE, null=True)
    symbol_removal = models.TextField(null=True)
    case_folding = models.TextField(null=True)
    tokenizing = models.TextField(null=True)
    slangword_removal = models.TextField(null=True)
    stopword_removal = models.TextField(null=True)
    stemming = models.TextField(null=True)
    text_bersih = models.TextField(null=True)
    sentimen = {
        ('positif', 'Positif'),
        ('negatif', 'Negatif'),
        ('netral', 'Netral'),
    }
    label = models.CharField(max_length=255, choices=sentimen, null=True)
    
    def __str__(self):
        return "Preprocessing"