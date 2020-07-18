
from django.shortcuts import render

from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required

from reports.dashapps import HistorialBalance

from .serializers import *
from .forms import *


# Create your views here.


@require_http_methods(["GET", "POST"])
@login_required(login_url='/accounts/login')
def categorias(request):
    if request.method == 'POST':
        categoria = Categoria()
        categoria.name = request.POST['name']
        categoria.key = request.POST['key']
        categoria.save()

    #categorias = [(i.name, i.id) for i in Categoria.objects.all()]
    categorias = CategoriaSerializer(Categoria.objects.all(), many=True)
    c = dict(categorias=categorias.data)

    return render(request, 'contable/categorias.html', c)


@require_http_methods(["GET", "POST"])
@login_required(login_url='/accounts/login')
def cuentas(request):
    tickets = Ticket.objects.all().order_by('fecha')
    HistorialBalance(tickets, convert=True)
    if request.method == 'POST':
        cuenta = Cuenta()
        cuenta.name = request.POST['name']
        cuenta.key = request.POST['key']
        cuenta.save()
    else:
        form = CuentaForm()

    cuentas = CuentaSerializer(Cuenta.objects.all(), many=True)
    c = dict(cuentas=cuentas.data, form=form)

    return render(request, 'contable/cuentas.html', c)


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

    return render(request, 'contable/modos_transferencia.html', c)
