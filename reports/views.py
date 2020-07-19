from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse, HttpResponseBadRequest
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required

from contable.serializers import *
from django.core.paginator import Paginator

import datetime
# Create your views here.


@require_http_methods(["GET"])
@login_required(login_url='/accounts/login')
def cuenta(request, cuenta_id=None):
    if cuenta_id:
        session = request.session
        bargraph_state = session.get('django_plotly_dash', {})
        bargraph_state['cuenta'] = cuenta_id
        session['django_plotly_dash'] = bargraph_state
        c = dict()
        return render(request, 'reports/cuenta.html', c)

    else:

        return redirect('cuentas/')


@require_http_methods(['GET'])
@login_required(login_url='accounts/login')
def get_tickets(request, cuenta_id=None, categoria_id=None):
    tickets = Ticket.objects.all().order_by('-fecha')
    if cuenta_id:
        cuenta = Cuenta.objects.get(id=cuenta_id)
        tickets = tickets.filter(cuenta=cuenta)
    if categoria_id:
        categoria = Categoria.objects.get(id=categoria_id)
        tickets = tickets.filter(categoria=categoria)

    paginator = Paginator(tickets, 15)
    page_number = request.GET.get('page')
    tickets = paginator.get_page(page_number)
    jtickets = TicketSerializer(tickets, many=True).data

    categorias = [cat.name for cat in Categoria.objects.all()]

    c = dict(
        tickets=jtickets,
        pages=list(range(1,paginator.num_pages)),
        page_number=page_number,
        categorias=categorias,
    )

    return JsonResponse(c)


@require_http_methods(['PUT'])
@login_required(login_url="/accounts/login")
def categorize(request):
    ticket_id = request.GET.get('ticket_id')
    categoria_id = request.GET.get('categoria_id')
    if ticket_id and categoria_id:
        ticket = Ticket.objects.get(id=ticket_id)
        ticket.categoria = Categoria.objects.get(id=categoria_id)
        ticket.save()
        return HttpResponse('')
    else:
        return HttpResponseBadRequest('specify categoria and cuenta')


@require_http_methods(["GET"])
@login_required(login_url='/accounts/login')
def categoria(request, categoria_id=None):
    if categoria_id:
        session = request.session
        bargraph_state = session.get('django_plotly_dash', {})
        bargraph_state['categoria'] = categoria_id
        session['django_plotly_dash'] = bargraph_state
        c = dict()
        return render(request, 'reports/categoria.html', c)

    else:

        return redirect('categorias/')
