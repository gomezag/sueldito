# Generated by Django 3.0.8 on 2020-09-23 10:43

import contable.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('contable', '0014_auto_20200922_1746'),
    ]

    operations = [
        migrations.AlterField(
            model_name='ticket',
            name='proyecto',
            field=models.ForeignKey(on_delete=models.SET(contable.models.get_unassigned), to='contable.Proyecto'),
        ),
    ]
