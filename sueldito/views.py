from django.shortcuts import render

from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required

from contable.serializers import *
from contable.models import *

@require_http_methods(["GET", "POST"])
@login_required(login_url='/accounts/login')
def home(request):
    c = get_base_context()
    return render(request, 'home.html', c)


def get_base_context():
    c = dict()

    c['cuentas'] = CuentaSerializer(Cuenta.objects.all().order_by('name'), many=True).data
    c['categorias'] = CategoriaSerializer(Categoria.objects.all().order_by('name'), many=True).data
    c['proyectos'] = ProyectoSerializer(Proyecto.objects.all().order_by('name'), many=True).data

    return c