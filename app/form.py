from django import forms
from .models import lidardataModel

class lidardataModelForm(forms.ModelForm):
    class Meta:
        model = lidardataModel
        fields=('poseImg','keypoints')