from django import forms
from .models import *
from colorfield.widgets import ColorWidget


class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField()


class CuentaForm(forms.ModelForm):

    moneda = forms.ModelChoiceField(queryset=Moneda.objects.all())
    class Meta:
        model = Cuenta
        fields = "__all__"


class CategoriaForm(forms.ModelForm):

    class Meta:
        model = Categoria
        fields = ['name', 'color']
        widgets = {
            'color': ColorWidget,
        }

class TicketForm(forms.ModelForm):
    fecha = forms.DateField()
    cuenta = forms.ModelChoiceField(queryset=Cuenta.objects.all())
    categoria = forms.ModelChoiceField(queryset=Categoria.objects.all())
    class Meta:
        model = Ticket
        fields = "__all__"


class ProyectoForm(forms.ModelForm):
    class Meta:
        model = Proyecto
        fields = ['name']

