import pandas
import datetime
from io import StringIO

from django.shortcuts import render
from django.http import HttpResponseRedirect

from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required

from contable.models import *

# Create your views here.


@require_http_methods(["POST"])
@login_required(login_url='/accounts/login')
def parse_file(request):
    c = dict()
    file = request.FILES['file']
    if file.multiple_chunks():
        return render(request, 'contable/parse_file.html', c)
    file = file.read()
    try:
        file_data = StringIO(file.decode("utf-8"))
    except UnicodeDecodeError:
        file_data = StringIO(file.decode("utf-16"))

    df = pandas.read_csv(file_data)
    c['columns'] = df.columns
    c['cuentas'] = [i.name for i in Cuenta.objects.all()]
    c['data'] = df.to_dict()
    c['jdata'] = df.to_json()
    c['monedas'] = [i.name for i in Moneda.objects.all()]
    return render(request, 'import/parse_file.html', c)


@require_http_methods(["POST"])
@login_required(login_url='/accounts/login')
def import_file(request):
    df = pandas.read_json(request.POST['data'])
    cuenta = Cuenta.objects.get(name=request.POST['cuenta'])
    moneda = Moneda.objects.get(name=request.POST['moneda'])
    tickets = []

    for i in range(0, len(df)):
        ticket = Ticket()
        ticket.concepto = df[request.POST['concepto']][i]
        importe = df[request.POST['importe']][i]
        if type(importe) == str:
            importe = float(importe.replace(',', '.'))
        ticket.importe = importe
        ticket.tipo = ModoTransferencia.objects.get(name=df[request.POST['modo']][i])
        ticket.fecha = datetime.datetime.strptime(df[request.POST['fecha']][i], request.POST['date_format']).date()
        ticket.moneda = moneda
        ticket.cuenta = cuenta
        ticket.categoria = Categoria.objects.get(name="No Categorizado")
        tickets.append(ticket)

    for ticket in tickets:
        ticket.save()

    return render(request, 'home.html', dict())


@require_http_methods(["POST"])
@login_required(login_url='/accounts/login')
def categorize(request):
    categoria = Categoria.objects.get(name=request.POST['categoria'])
    ticket_ids = request.POST.getlist('tickets')
    tickets = Ticket.objects.filter(pk__in=ticket_ids)
    print(tickets)
    cuenta = Cuenta.objects.get(name=request.POST['cuenta'])

    match_ids = request.POST.getlist('matchs')

    for id in ticket_ids:
        ticket = Ticket.objects.get(id=id)
        ticket.categoria = categoria
        ticket.save()

    for id in match_ids:
        concepto = Ticket.objects.get(id=id).concepto
        for ticket in Ticket.objects.filter(concepto=concepto):
            ticket.categoria = categoria
            ticket.save()

    cuenta_id = cuenta.id

    return HttpResponseRedirect('/view/cuenta/'+str(cuenta_id))
