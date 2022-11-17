# Generated by Django 4.1.1 on 2022-10-18 02:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0003_userimgmodel'),
    ]

    operations = [
        migrations.CreateModel(
            name='bgRemovedImgModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='bgRemovedImg/')),
            ],
        ),
        migrations.AlterField(
            model_name='userimgmodel',
            name='image',
            field=models.ImageField(upload_to='UserImg/'),
        ),
    ]