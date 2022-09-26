from django.contrib import admin
from .models import lidardataModel

# Register your models here.
class lidardataAdmin(admin.ModelAdmin):
    list_display = ('poseImg','keypoints')
    
admin.site.register(lidardataModel,lidardataAdmin)