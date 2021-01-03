from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse, HttpResponseBadRequest
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required

from rest_framework.parsers import JSONParser
from django.core.paginator import Paginator

from contable.serializers import *
from contable.forms import *
from contable.filters import *
from sueldito.views import get_base_context
from . import dashapps


import datetime



@require_http_methods(['GET'])
@login_required(login_url='accounts/login')
def get_tickets(request, cuenta_id=None, categoria_id=None, proyecto_id=None, fechas=None):
    tickets = Ticket.objects.all().order_by('-fecha')
    if cuenta_id:
        cuenta = Cuenta.objects.get(id=cuenta_id)
        tickets = tickets.filter(cuenta=cuenta)
    if categoria_id:
        categoria = Categoria.objects.get(id=categoria_id)
        tickets = tickets.filter(categoria=categoria)
    if proyecto_id:
        proyecto = Proyecto.objects.get(id=proyecto_id)
        tickets = tickets.filter(proyecto=proyecto)
    if request.GET.get('keyword'):
        keyword = request.GET.get('keyword')
        tickets = tickets.filter(concepto__icontains=keyword)
    if request.GET.get('categoria'):
        categoria = Categoria.objects.get(id=request.GET.get('categoria'))
        tickets = tickets.filter(categoria=categoria)
    if fechas:
        tickets = tickets.filter(fecha__gte=fechas[0], fecha__lte=fechas[1])
    total_tickets = [ticket.id for ticket in tickets]
    paginator = Paginator(tickets, 15)
    page_number = request.GET.get('page')
    tickets = paginator.get_page(page_number)
    jtickets = TicketSerializer(tickets, many=True).data

    c = dict(
        tickets=jtickets,
        total_tickets=total_tickets,
        pages=list(range(1, paginator.num_pages+1)),
        page_number=page_number,
    )
    c = {**c, **get_base_context()}
    return JsonResponse(c)


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