from django import forms
from .models import *


class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField()


class CuentaForm(forms.ModelForm):
    class Meta:
        model = Cuenta
        fields = "__all__"


class CategoriaForm(forms.ModelForm):
    class Meta:
        model = Cuenta
        fields = "__all__"
