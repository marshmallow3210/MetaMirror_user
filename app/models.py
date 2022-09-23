from django.db import models

# Create your models here.
class Modelimg(models.Model):
    #衣服圖片
    img = models.ImageField()
    # upload_date = models.DateField(default=timezone.now)