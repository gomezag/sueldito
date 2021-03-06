import pandas
import datetime
from io import StringIO

from django.shortcuts import render
from django.http import HttpResponseRedirect

from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required

from contable.serializers import *
from .forms import ImportFileForm

@require_http_methods(["GET"])
@login_required(login_url='/accounts/login')
def import_view(request):
    c = dict()
    c['cuentas'] = CuentaSerializer(Cuenta.objects.all(), many=True).data
    c['categorias'] = CategoriaSerializer(Categoria.objects.all(), many=True).data
    c['proyectos'] = ProyectoSerializer(Proyecto.objects.all(), many=True).data

    c['form'] = ImportFileForm()
    return render(request, 'import/import.html', c)


@require_http_methods(["POST"])
@login_required(login_url='/accounts/login')
def parse_file(request):
    c = dict()
    c['cuentas'] = CuentaSerializer(Cuenta.objects.all(), many=True).data
    c['categorias'] = CategoriaSerializer(Categoria.objects.all(), many=True).data
    c['proyectos'] = ProyectoSerializer(Proyecto.objects.all(), many=True).data
    form = ImportFileForm(request.POST, request.FILES)
    file = request.FILES['file']
    print(file)
    file_data = file.read()
    print(request.POST['type'])
    if request.POST['type'] == 'bbvaes':
        df = pandas.read_excel(file_data, usecols="B:J", skiprows=range(1,4), header=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.iloc[pandas.RangeIndex(len(df)).drop(0)]
        df = df.reset_index()
        df = df.dropna(subset=['Fecha'])
        modos = []

        if 'Movimiento' in df.columns:
            for i in range(len(df)):
                if 'Transferencia' in df['Concepto'][i]:
                    modos.append('Transferencia')
                    df['Concepto'][i] = df['Movimiento'][i]
                elif 'Adeudo' in df['Movimiento'][i]:
                    modos.append('Debito Automatico')
                elif 'Ret. efectivo' in df['Concepto'][i]:
                    modos.append('Efectivo')
                else:
                    modos.append('Debito')
        else:
            modos=['Credito' for i in range(len(df))]

        df['Modo'] = pandas.Series(modos, index=df.index)

        df = pandas.concat([df['Fecha'], df['Importe'], df['Concepto'], df['Modo']], axis=1,
                       keys=['Fecha', 'Importe', 'Concepto', 'Modo'])

    elif request.POST['type'] == 'bbvapy':
        df = pandas.read_excel(file_data)
        df = df.iloc[12:,1:]
        df = df.rename(columns=df.iloc[0])
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.iloc[pandas.RangeIndex(len(df)).drop(0)]
        df = df.reset_index()
        df = df.drop(columns=['index'])
        first_null = df[df[df.columns[1]].isnull()].index.to_list()[0]
        df = df.iloc[:first_null, :]
        modos = []
        importes = []
        for i in range(len(df)):
            if 'TRANSFERENCIA' in df['Concepto'][i]:
                modos.append('Transferencia')
            elif 'DEPOSITO' in df['Concepto'][i]:
                modos.append('Deposito')
            else:
                modos.append('Transferencia')
            total = float(df['Importe Crédito'][i].replace('.','').replace(',','.'))\
                    -float(df['Importe Débito'][i].replace('.','').replace(',','.'))
            importes.append(total)

        df['Modo'] = pandas.Series(modos, index=df.index)
        df['Importe'] = pandas.Series(importes, index=df.index)

        df = pandas.concat([df['Fecha de Transacción'], df['Importe'], df['Concepto'], df['Modo']], axis=1,
                       keys=['Fecha', 'Importe', 'Concepto', 'Modo'])

    elif request.POST['type'] == 'csv':
        try:
            df = pandas.read_csv(file_data, sep=",")
        except UnicodeDecodeError:
            df = pandas.read_csv(StringIO(file_data.read().decode("utf-16")))

    elif request.POST['type'] == 'bbvapycred':
        df = leer_excel_bbvapycred(file_data)

    else:
        return HttpResponseRedirect('error.html')
    c['columns'] = df.columns
    c['cuentas'] = [i.name for i in Cuenta.objects.all()]
    c['data'] = df.to_dict()
    c['jdata'] = df.to_json()
    c['monedas'] = [i.name for i in Moneda.objects.all()]
    return render(request, 'import/parse_file.html', c)


@require_http_methods(["POST"])
@login_required(login_url='/accounts/login')
def import_file(request):
    c = dict()
    c['cuentas'] = CuentaSerializer(Cuenta.objects.all(), many=True).data
    c['categorias'] = CategoriaSerializer(Categoria.objects.all(), many=True).data
    c['proyectos'] = ProyectoSerializer(Proyecto.objects.all(), many=True).data
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
        ticket.modo = ModoTransferencia.objects.get(name=df[request.POST['modo']][i])
        try:
            ticket.fecha = datetime.datetime.strptime(df[request.POST['fecha']][i], request.POST['date_format']).date()
        except TypeError:
            ticket.fecha = datetime.datetime.utcfromtimestamp(df[request.POST['fecha']][i]/1000).date()
        ticket.moneda = moneda
        ticket.cuenta = cuenta
        ticket.categoria = Categoria.objects.get(name="No Categorizado")
        ticket.proyecto = Proyecto.objects.get(name="No asignado")
        tickets.append(ticket)

    for ticket in tickets:
        ticket.save()

    return render(request, 'home.html', c)


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


def leer_excel_bbvapycred(fn):
    import pandas
    df = pandas.read_excel(fn, thousands='.')
    df = df.iloc[13:, 1:]
    df = df.rename(columns=df.iloc[0])
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.iloc[pandas.RangeIndex(len(df)).drop(0)]
    df = df.reset_index()
    df = df.drop(columns=['index'])
    first_null = df[df[df.columns[4]]=='Total Créditos:'].index.to_list()[0]
    df = df.iloc[1:first_null, :]
    df['Importe'] = pandas.to_numeric(df['Importe'], downcast='float')
    df['Importe'] = -df['Importe']
    r = pandas.DataFrame(columns=df.columns)
    for index, row in df.iterrows():
        if pandas.isna(row[0]):
            r.iloc[-1,3] += row[3]
            r.iloc[-1,5] += row[5]
        else:
            r = r.append(row)
    r['Modo'] = ['Credito' for i, row in r.iterrows()]
    r = r.reset_index()
    r = r.drop(columns=['index'])
    df = pandas.concat([r['Fecha de Operación'], r['Importe'], r['Detalle'], r['Modo']], axis=1,
                       keys=['Fecha', 'Importe', 'Concepto', 'Modo'])
    return df

