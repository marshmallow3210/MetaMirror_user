# Generated by Django 4.1.1 on 2022-09-26 18:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='bodyDataModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('shoulderWidth', models.TextField()),
                ('chestWidth', models.TextField()),
                ('clothingLength', models.TextField()),
            ],
        ),
    ]
