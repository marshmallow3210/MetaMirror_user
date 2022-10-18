from django.contrib import admin
from .models import lidardataModel,bodyDataModel,UserImgModel,bgRemovedImgModel

# Register your models here.
class lidardataAdmin(admin.ModelAdmin):
    list_display = ('poseImg','keypoints')
    
class bodyDataAdmin(admin.ModelAdmin):
    list_display = ('shoulderWidth','chestWidth','clothingLength')
    
class UserImgAdmin(admin.ModelAdmin):
    list_display = ('image',)
    
class bgRemovedImgAdmin(admin.ModelAdmin):
    list_display = ('image',)
    
admin.site.register(lidardataModel,lidardataAdmin)
admin.site.register(bodyDataModel,bodyDataAdmin)
admin.site.register(UserImgModel,UserImgAdmin)
admin.site.register(bgRemovedImgModel,bgRemovedImgAdmin)