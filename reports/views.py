from django.shortcuts import render
from django.core.paginator import Paginator
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required

from .dashapps import HistorialBalance
from contable.serializers import *

import datetime
# Create your views here.

@require_http_methods(["GET"])
@login_required(login_url='/accounts/login')
def cuenta(request, cuenta_id=None):
    if cuenta_id:
        cuenta = Cuenta.objects.get(id=cuenta_id)
        date_start = request.GET.get('date_start')
        date_end = request.GET.get('date_end')

        if date_start is None or date_end is None:
            date_start = datetime.date(2020,1,1)
            date_end = datetime.datetime.now().date()

        tickets = cuenta.ticket_set.filter(fecha__lte=date_end, fecha__gte=date_start).order_by('-fecha')

        HistorialBalance(tickets, convert=False)
        jtickets = TicketSerializer(tickets, many=True).data
        categorias = [i.name for i in Categoria.objects.all()]

        c = dict(
            tickets=jtickets,
            cuenta=cuenta.name,
            categorias=categorias)

    return render(request, 'reports/cuenta.html', c)


@require_http_methods(["GET"])
@login_required(login_url='/accounts/login')
def categoria(request, id=None):
    if id:
        cat = Categoria.objects.get(id=id)

        tickets = cat.ticket_set.all().order_by('-fecha')

        HistorialBalance(tickets, convert=False)
        jtickets = TicketSerializer(tickets, many=True).data

        categorias = [i.name for i in Categoria.objects.all()]

        c = dict(tickets=jtickets, categoria=cat.name, categorias=categorias)

    return render(request, 'reports/categoria.html', c)
