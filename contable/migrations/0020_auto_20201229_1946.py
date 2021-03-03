# Generated by Django 3.1.4 on 2020-12-29 19:46

import contable.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('contable', '0019_auto_20201229_1946'),
    ]

    operations = [
        migrations.AlterField(
            model_name='ticket',
            name='proyecto',
            field=models.ForeignKey(default=contable.models.get_unassigned, on_delete=models.SET(contable.models.get_unassigned), to='contable.proyecto'),
        ),
    ]
