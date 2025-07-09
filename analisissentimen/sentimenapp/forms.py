from django import forms
from .models import Stopword
from .models import Slangword

class StopwordForm(forms.ModelForm):
    class Meta:
        model = Stopword
        fields = ['stopwords']

class SlangwordForm(forms.ModelForm):
    class Meta:
        model = Slangword
        fields = ['katabaku', 'katatidakbaku']
