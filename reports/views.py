from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required

from contable.serializers import *
from contable.forms import *
from contable.filters import *
from sueldito.views import get_base_context

from .dashapps import *

@require_http_methods(["GET", "PUT", "DELETE"])
@login_required(login_url='/accounts/login')
def cuenta(request, cuenta_id=None):
    if cuenta_id:
        if request.method == 'GET':
            session = request.session
            bargraph_state = session.get('django_plotly_dash', {})
            bargraph_state['cuenta'] = cuenta_id
            bargraph_state['categoria'] = None
            bargraph_state['proyecto'] = None
            session['django_plotly_dash'] = bargraph_state

            c = get_base_context()
            c['cuenta'] = CuentaSerializer(Cuenta.objects.get(id=cuenta_id)).data
            c['ticketForm'] = TicketForm()
            c['filter'] = TicketFilter(request.GET, queryset=Ticket.objects.all())
            return render(request, 'reports/cuenta.html', c)

    else:

        return redirect('cuentas/')


@require_http_methods(["GET"])
@login_required(login_url='/accounts/login')
def proyecto(request, proyecto_id=None):
    if proyecto_id:
        session = request.session
        bargraph_state = session.get('django_plotly_dash', {})
        bargraph_state['cuenta'] = None
        bargraph_state['categoria'] = None
        bargraph_state['proyecto'] = proyecto_id
        session['django_plotly_dash'] = bargraph_state

        c = get_base_context()
        c['proyecto'] = ProyectoSerializer(Proyecto.objects.get(id=proyecto_id)).data

        return render(request, 'reports/proyecto.html', c)

    else:

        return redirect('proyectos/')

@require_http_methods(["GET"])
@login_required(login_url='/accounts/login')
def categoria(request, categoria_id=None):
    if categoria_id:
        session = request.session
        bargraph_state = session.get('django_plotly_dash', {})
        bargraph_state['cuenta'] = None
        bargraph_state['categoria'] = categoria_id
        bargraph_state['proyecto'] = None
        session['django_plotly_dash'] = bargraph_state

        c = get_base_context()
        c['categoria'] = CategoriaSerializer(Categoria.objects.get(id=categoria_id)).data

        return render(request, 'reports/categoria.html', c)

    else:

        return redirect('categorias/')


@require_http_methods(["GET"])
@login_required(login_url='/accounts/login')
def consulta(request):
        session = request.session
        bargraph_state = session.get('django_plotly_dash', {})
        bargraph_state['cuenta'] = None
        bargraph_state['categoria'] = None
        bargraph_state['proyecto'] = None
        session['django_plotly_dash'] = bargraph_state
        c = get_base_context()

        return render(request, 'reports/consulta.html', c)