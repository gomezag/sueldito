
from django.shortcuts import render

from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required

# from reports.dashapps import HistorialBalance

from .serializers import *
from .forms import *


# Create your views here.
@require_http_methods(["GET", "POST"])
@login_required(login_url='/accounts/login')
def home(request):
    c = dict()
    c['cuentas'] = CuentaSerializer(Cuenta.objects.all(), many=True).data
    c['categorias'] = CategoriaSerializer(Categoria.objects.all(), many=True).data
    c['proyectos'] = ProyectoSerializer(Proyecto.objects.all(), many=True).data
    return render(request, 'home.html', c)

@require_http_methods(["GET", "POST"])
@login_required(login_url='/accounts/login')
def categorias(request):
    if request.method == 'POST':
        categoria = Categoria()
        categoria.name = request.POST['name']
        categoria.key = request.POST['key']
        categoria.save()

    elif request.method == "PUT":
        if request.PUT.get("cat_id") and request.PUT.get("new_name"):
            categoria = Categoria.objects.get(id=request.PUT.get("cat_id"))
            categoria.name = request.PUT.get("new_name")
            categoria.save()

    c = dict()
    c['cuentas'] = CuentaSerializer(Cuenta.objects.all(), many=True).data
    c['categorias'] = CategoriaSerializer(Categoria.objects.all(), many=True).data
    c['proyectos'] = ProyectoSerializer(Proyecto.objects.all(), many=True).data
    return render(request, 'contable/categorias.html', c)


@require_http_methods(["GET", "POST"])
@login_required(login_url='/accounts/login')
def cuentas(request):
    session = request.session
    bargraph_state = session.get('django_plotly_dash', {})
    bargraph_state['cuenta'] = 'all'
    bargraph_state['categoria'] = None
    session['django_plotly_dash'] = bargraph_state

    if request.method == 'POST':
        cuenta = Cuenta()
        cuenta.name = request.POST['name']
        cuenta.key = request.POST['key']
        cuenta.save()
    else:
        form = CuentaForm()

    c = dict(form=form)
    c['cuentas'] = CuentaSerializer(Cuenta.objects.all(), many=True).data
    c['categorias'] = CategoriaSerializer(Categoria.objects.all(), many=True).data
    c['proyectos'] = ProyectoSerializer(Proyecto.objects.all(), many=True).data
    return render(request, 'contable/cuentas.html', c)

@require_http_methods(["GET", "POST"])
@login_required(login_url='/accounts/login')
def proyectos(request):
    if request.method == 'POST':
        proyecto = Proyecto()
        proyecto.name = request.POST['name']
        proyecto.save()
        c=dict()
    else:
        form = ProyectoForm()
        c = dict(form=form)


    c['cuentas'] = CuentaSerializer(Cuenta.objects.all(), many=True).data
    c['categorias'] = CategoriaSerializer(Categoria.objects.all(), many=True).data
    c['proyectos'] = ProyectoSerializer(Proyecto.objects.all(), many=True).data
    return render(request, 'contable/proyectos.html', c)


@require_http_methods(["GET", "POST"])
def monedas(request):
    if request.method == 'POST':
        moneda = Moneda()
        moneda.name = request.POST['name']
        moneda.key = request.POST['key']
        moneda.cambio = request.POST['cambio']
        moneda.save()

    monedas = [i.name for i in Moneda.objects.all()]
    c = dict(monedas=monedas)
    c['cuentas'] = CuentaSerializer(Cuenta.objects.all(), many=True).data
    c['categorias'] = CategoriaSerializer(Categoria.objects.all(), many=True).data
    c['proyectos'] = ProyectoSerializer(Proyecto.objects.all(), many=True).data
    return render(request, 'contable/monedas.html', c)


@require_http_methods(["GET", "POST"])
def modos_transferencia(request):
    if request.method == 'POST':
        modo_transferencia = ModoTransferencia()
        modo_transferencia.name = request.POST['name']
        modo_transferencia.key = request.POST['key']
        modo_transferencia.save()

    modos = [i.name for i in ModoTransferencia.objects.all()]
    c = dict(modos=modos)
    c['cuentas'] = CuentaSerializer(Cuenta.objects.all(), many=True).data
    c['categorias'] = CategoriaSerializer(Categoria.objects.all(), many=True).data
    c['proyectos'] = ProyectoSerializer(Proyecto.objects.all(), many=True).data
    return render(request, 'contable/modos_transferencia.html', c)
