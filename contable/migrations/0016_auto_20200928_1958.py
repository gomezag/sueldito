# Generated by Django 3.0.8 on 2020-09-28 19:58

import contable.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('contable', '0015_auto_20200923_1043'),
    ]

    operations = [
        migrations.AlterField(
            model_name='ticket',
            name='categoria',
            field=models.ForeignKey(default=models.SET(contable.models.get_uncategorized), on_delete=models.SET(contable.models.get_uncategorized), to='contable.Categoria'),
        ),
        migrations.AlterField(
            model_name='ticket',
            name='proyecto',
            field=models.ForeignKey(default=models.SET(contable.models.get_unassigned), on_delete=models.SET(contable.models.get_unassigned), to='contable.Proyecto'),
        ),
    ]