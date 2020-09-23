from django.shortcuts import render

from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required

from contable.serializers import *
from contable.models import *

@require_http_methods(["GET", "POST"])
@login_required(login_url='/accounts/login')
def home(request):
    c = dict()
    c['cuentas'] = CuentaSerializer(Cuenta.objects.all(), many=True).data
    c['categorias'] = CategoriaSerializer(Categoria.objects.all(), many=True).data
    c['proyectos'] = ProyectoSerializer(Proyecto.objects.all(), many=True).data
    return render(request, 'home.html', c)

