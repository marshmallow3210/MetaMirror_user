from django import forms
from .models import Modelimg

class ModelForm(forms.ModelForm):
    class Meta:
        model = Modelimg
        fields=('image',)
        widgets={
            'image': forms.FileInput(attrs={'class': 'form-control-file'})
        }