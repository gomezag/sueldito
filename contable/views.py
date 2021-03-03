
from django.shortcuts import render

from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse, HttpResponseBadRequest

from sueldito.views import get_base_context

# from reports.dashapps import HistorialBalance

from .serializers import *
from .forms import *

from django.core.paginator import Paginator

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
        f = CategoriaForm(request.POST)
        if f.is_valid():
            f.save()
        else:
            return render(request, 'error.html', dict(error='bad form'))

    elif request.method == "PUT":
        if request.PUT.get("cat_id") and request.PUT.get("new_name"):
            categoria = Categoria.objects.get(id=request.PUT.get("cat_id"))
            categoria.name = request.PUT.get("new_name")
            categoria.save()

    c = dict()
    c['cuentas'] = CuentaSerializer(Cuenta.objects.all(), many=True).data
    c['categorias'] = CategoriaSerializer(Categoria.objects.all(), many=True).data
    c['proyectos'] = ProyectoSerializer(Proyecto.objects.all(), many=True).data
    c['form'] = CategoriaForm()
    return render(request, 'contable/categorias.html', c)


@require_http_methods(["GET", "POST"])
@login_required(login_url='/accounts/login')
def cuentas(request, cuenta_id=None):
    if cuenta_id is None:
        session = request.session
        bargraph_state = session.get('django_plotly_dash', {})
        bargraph_state['cuenta'] = 'all'
        bargraph_state['categoria'] = None
        session['django_plotly_dash'] = bargraph_state

        if request.method == 'POST':
            f = CuentaForm(request.POST)
            if f.is_valid():
                f.save()

        form = CuentaForm()

        c = dict(form=form)
        c = {**c, **get_base_context()}
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
@login_required(login_url='/accounts/login')
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


@require_http_methods(["GET", "POST", "PUT"])
@login_required(login_url='accounts/login')
def activos(request):
    if request.method == "POST":
        if 'id' in request.POST.keys():
            activo = Activo.objects.get(id=request.POST['id'])
            form = ActivoForm(request.POST, instance=activo)
        else:
            form = ActivoForm(request.POST)
            if form.is_valid():
                activo = Activo(**form.cleaned_data)
        if form.is_valid():
            #print(activo.name, activo.id)
            activo.save()
    c = get_base_context()
    if 'form_id' in request.GET.keys():
        activo = ActivoSerializer(Activo.objects.get(id=request.GET['form_id'])).data
        cform = ActivoForm(initial=activo)
        c['form'] = cform
        c['id'] = activo['id']
        return render(request, 'contable/activoform.html', c)
    else:
        c['form'] = ActivoForm()
        c['activos'] = ActivoSerializer(Activo.objects.all(), many=True).data
        return render(request, 'contable/activos.html', c)

