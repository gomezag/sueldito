from django.shortcuts import render

from sueldito.views import get_base_context
from contable.serializers import *
import pandas
import datetime

from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required

from .nn import AutoClassifier
# Create your views here.

@require_http_methods(["POST"])
@login_required(login_url='/accounts/login')
def auto_classify(request):
    c = get_base_context()
    df = pandas.read_json(request.POST['data'])
    classifier = AutoClassifier()
    classifier.load_nn('./torch_models/minibatch')
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
        ticket.modo = ModoTransferencia.objects.get(name=df[request.POST['modo']][i])
        try:
            ticket.fecha = datetime.datetime.strptime(df[request.POST['fecha']][i], request.POST['date_format']).date()
        except TypeError:
            ticket.fecha = datetime.datetime.utcfromtimestamp(df[request.POST['fecha']][i]/1000).date()
        ticket.moneda = moneda
        ticket.cuenta = cuenta
        ticket.categoria = classifier.predict(ticket, 0.6) # Categoria.objects.get(name="No Categorizado")
        ticket.proyecto = Proyecto.objects.get(name="No asignado")
        tickets.append(ticket)

    dicto = TicketSerializer(tickets, many=True).data
    for item in dicto:
        item['moneda'] = item['moneda']['key']
        item['categoria'] = item['categoria']['name']
        item['proyecto'] = item['proyecto']['name']
    df = pandas.DataFrame.from_dict(dicto)

    c['data'] = df.to_dict()
    c['tickets'] = TicketSerializer(tickets, many=True).data
    return render(request, 'import/confirm-classification.html', c)