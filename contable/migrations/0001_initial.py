# Generated by Django 3.0.6 on 2020-05-24 16:12

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Categoria',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('key', models.CharField(max_length=5)),
                ('name', models.CharField(max_length=200)),
                ('color', models.CharField(max_length=10, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Cuenta',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=200)),
                ('balance', models.FloatField()),
                ('key', models.CharField(default='NONE', max_length=16)),
            ],
        ),
        migrations.CreateModel(
            name='ModoTransferencia',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=15)),
                ('key', models.CharField(max_length=2)),
            ],
        ),
        migrations.CreateModel(
            name='Moneda',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=20)),
                ('key', models.CharField(max_length=3)),
                ('cambio', models.FloatField(default=1)),
            ],
        ),
        migrations.CreateModel(
            name='Ticket',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('fecha', models.DateField(verbose_name='Fecha')),
                ('importe', models.FloatField(verbose_name='importe')),
                ('concepto', models.CharField(max_length=200)),
                ('consistency', models.BooleanField(default=False)),
                ('categoria', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='contable.Categoria')),
                ('cuenta', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='contable.Cuenta')),
                ('moneda', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='contable.Moneda')),
                ('tipo', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='contable.ModoTransferencia')),
            ],
        ),
        migrations.CreateModel(
            name='SubTicket',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('concepto', models.CharField(max_length=200)),
                ('importe', models.FloatField(verbose_name='importe')),
                ('categoria', models.CharField(default='Total', max_length=200)),
                ('ticket', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='contable.Ticket')),
                ('transaccion', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='contable.ModoTransferencia')),
            ],
        ),
    ]
