from django import forms
from .models import lidardataModel,bodyDataModel,UserImgModel

class lidardataModelForm(forms.ModelForm):
    class Meta:
        model = lidardataModel
        fields=('poseImg','keypoints')
        
class bodyDataModelForm(forms.ModelForm):
    class Meta:
        model = bodyDataModel
        fields=('shoulderWidth','chestWidth','clothingLength')

class UserImgModelForm(forms.ModelForm):
    class Meta:
        model = UserImgModel
        fields=('image',)