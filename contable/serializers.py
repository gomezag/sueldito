from rest_framework import serializers
from .models import *


class MonedaSerializer(serializers.ModelSerializer):

    class Meta:
        model = Moneda
        fields = "__all__"


class CuentaSerializer(serializers.ModelSerializer):
    # balance = serializers.ReadOnlyField()
    balance = serializers.SerializerMethodField()
    moneda = MonedaSerializer()

    def get_balance(self, obj):
        return "{:,.2f}".format(obj.balance)

    class Meta:
        model = Cuenta
        fields = "__all__"


class CategoriaSerializer(serializers.ModelSerializer):

    class Meta:
        model = Categoria
        fields = "__all__"


class TicketSerializer(serializers.ModelSerializer):
    importe = serializers.SerializerMethodField()
    moneda = MonedaSerializer()
    categoria = CategoriaSerializer()
    saldo = serializers.SerializerMethodField()

    def get_saldo(self, obj):
        return "{:,.2f}".format(obj.saldo)

    def get_importe(self, obj):
        return "{:,.2f}".format(obj.importe)

    class Meta:
        model = Ticket
        fields = "__all__"