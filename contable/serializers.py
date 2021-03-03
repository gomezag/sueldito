from rest_framework import serializers
from .models import *


class MonedaSerializer(serializers.ModelSerializer):

    class Meta:
        model = Moneda
        fields = "__all__"


class ProyectoSerializer(serializers.ModelSerializer):

    class Meta:
        model = Proyecto
        fields = '__all__'


class CuentaSerializer(serializers.ModelSerializer):
    # balance = serializers.ReadOnlyField()
    balance = serializers.SerializerMethodField()
    moneda = MonedaSerializer()

    def get_balance(self, obj):
        return obj.balance

    class Meta:
        model = Cuenta
        fields = "__all__"

class ActivoSerializer(serializers.ModelSerializer):
    valor = serializers.SerializerMethodField()
    ganancias = serializers.SerializerMethodField()
    moneda = MonedaSerializer()

    def get_valor(self, obj):
        return obj.valor

    def get_ganancias(self, obj):
        return obj.ganancias
    class Meta:
        model = Activo
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
    proyecto = ProyectoSerializer()
    cuenta_destino = CuentaSerializer()
    cuenta = CuentaSerializer()
    activo = ActivoSerializer()

    def get_saldo(self, obj):
        return "{:,.2f}".format(obj.saldo)

    def get_importe(self, obj):
        return "{:,.2f}".format(obj.importe)

    class Meta:
        model = Ticket
        fields = "__all__"


class ReadOnlyTicketSerializer(serializers.ModelSerializer):
    importe = serializers.SerializerMethodField()
    moneda = serializers.SlugRelatedField(slug_field='key', read_only=True)
    categoria = serializers.SlugRelatedField(slug_field='name', read_only=True)
    saldo = serializers.SerializerMethodField()
    proyecto = serializers.SlugRelatedField(slug_field='name', read_only=True)
    cuenta_destino = serializers.SlugRelatedField(slug_field='name', read_only=True)
    cuenta = serializers.SlugRelatedField(slug_field='name', read_only=True)
    activo = serializers.SlugRelatedField(slug_field='name', read_only=True)

    def get_saldo(self, obj):
        return "{:,.2f}".format(obj.saldo)

    def get_importe(self, obj):
        return "{:,.2f}".format(obj.importe)

    class Meta:
        model = Ticket
        fields = "__all__"

