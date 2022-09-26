from django.contrib import admin
from .models import lidardataModel,bodyDataModel

# Register your models here.
class lidardataAdmin(admin.ModelAdmin):
    list_display = ('poseImg','keypoints')
    
class bodyDataAdmin(admin.ModelAdmin):
    list_display = ('shoulderWidth','chestWidth','clothingLength')
    
admin.site.register(lidardataModel,lidardataAdmin)
admin.site.register(bodyDataModel,bodyDataAdmin)