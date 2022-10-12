from django.urls import path,re_path
from . import views

urlpatterns = [
    path('openLidar',views.openLidar,name='openLidar'), # Lidar影像畫面
]