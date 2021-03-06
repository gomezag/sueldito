# Generated by Django 3.1.4 on 2020-12-29 19:46

import contable.models
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('contable', '0018_auto_20201127_2130'),
    ]

    operations = [
        migrations.AddField(
            model_name='ticket',
            name='cuenta_destino',
            field=models.ForeignKey(blank=True, default=None, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='CuentaDestino', to='contable.cuenta'),
        ),
        migrations.AlterField(
            model_name='ticket',
            name='categoria',
            field=models.ForeignKey(on_delete=models.SET(contable.models.get_uncategorized), to='contable.categoria'),
        ),
        migrations.AlterField(
            model_name='ticket',
            name='proyecto',
            field=models.ForeignKey(on_delete=models.SET(contable.models.get_unassigned), to='contable.proyecto'),
        ),
    ]
