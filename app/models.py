from django.db import models

class lidardataModel(models.Model):
    poseImg = models.TextField()
    keypoints = models.TextField()
    
class bodyDataModel(models.Model):
    shoulderWidth = models.TextField()
    chestWidth = models.TextField()
    clothingLength = models.TextField()