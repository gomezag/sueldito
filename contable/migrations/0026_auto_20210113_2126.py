# Generated by Django 3.1.4 on 2021-01-13 21:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('contable', '0025_auto_20201230_2229'),
    ]

    operations = [
        migrations.RenameModel(
            old_name='Inversion',
            new_name='Activo',
        ),
        migrations.AlterField(
            model_name='ticket',
            name='tax',
            field=models.CharField(blank=True, choices=[('IRPF', 'IRPF'), ('IVA', 'IVA'), ('DEDU', 'Deducible'), (None, 'None')], max_length=4, null=True),
        ),
    ]
