# Generated by Django 3.0.6 on 2020-07-16 19:04

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('contable', '0004_auto_20200716_1844'),
    ]

    operations = [
        migrations.AddField(
            model_name='cuenta',
            name='moneda',
            field=models.ForeignKey(default=1, on_delete=django.db.models.deletion.PROTECT, to='contable.Moneda'),
            preserve_default=False,
        ),
    ]
