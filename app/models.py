from django.db import models

class lidardataModel(models.Model):
    poseImg = models.TextField()
    keypoints = models.TextField()