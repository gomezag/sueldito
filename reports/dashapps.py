import datetime

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
from django_plotly_dash import DjangoDash

from django.db.models import Sum, F
from django.db.models.functions import TruncMonth, TruncYear
from dash.dependencies import Input, Output


historial = DjangoDash('BalanceHistorial')
graphs = []

historial.layout = html.Div([
    dcc.DatePickerRange(
        id='date-picker',
        start_date = (datetime.date.today()-datetime.timedelta(days=90)).replace(day=1),
        end_date = datetime.date.today(),
    ),
    dcc.RadioItems(
        id='convert',
        options=[
            {'label': 'In euros', 'value': 'EUR'},
            {'label': 'In local curr.', 'value': 'LOC'}
        ],
        value='LOC',
        labelStyle={'display': 'inline-block'}
    ),
    html.Div(id='plotarea', children=graphs),
])

@historial.expanded_callback(
    Output('plotarea', 'children'),
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('convert', 'value')]
)
def update_graph(start_date, end_date, convert, session_state=None, *args, **kwargs):
    import contable.models as md
    if session_state is None:
        raise NotImplemented('Session not created')
    cuenta = session_state.get('cuenta', None)
    categoria = session_state.get('categoria', None)
    proyecto = session_state.get('proyecto', None)
    tickets = md.Ticket.objects.filter(fecha__gte=start_date, fecha__lte=end_date)
    if categoria:
        tickets = tickets.filter(categoria=md.Categoria.objects.get(id=categoria))
    if proyecto:
        tickets = tickets.filter(proyecto=md.Proyecto.objects.get(id=proyecto))
    if cuenta and cuenta != 'all':
        tickets = tickets.filter(cuenta=md.Cuenta.objects.get(id=cuenta))
    else:
        convert = 'EUR'
    tickets = tickets.order_by('fecha')
    if convert == 'EUR':
        tickets = tickets.annotate(importe_conv=F('importe') * F('moneda__cambio'))
    else:
        tickets = tickets.annotate(importe_conv=F('importe'))
    tickets = tickets.annotate(month=TruncMonth('fecha'))
    tickets = tickets.annotate(categoria_name=F('categoria__name'))

    dfbars = pd.DataFrame(list(tickets.values('month', 'categoria_name').order_by(
        'month').annotate(sum=Sum('importe_conv'))))

    dfscat = pd.DataFrame(list(tickets.values('month').order_by(
        'month').annotate(sum=Sum('importe_conv'))))

    dfpies = pd.DataFrame(list(tickets.values('categoria_name').order_by(
        'month').annotate(sum=-Sum('importe_conv'))))
    savings = sum([ticket.importe_conv for ticket in tickets])
    if savings > 0:
        dfpies = dfpies.append(pd.DataFrame(columns=dfpies.columns, data=[('Savings', savings)]))

    colors = dict()
    for cat in md.Categoria.objects.all():
        colors[cat.name]=cat.color
    colors['Savings'] = "#006600"

    if len(tickets) > 1:

        bars = px.bar(dfbars,
                      x='month',
                      y='sum',
                      color='categoria_name',
                      color_discrete_map=colors,
                      labels={
                          "categoria_name": "Categoria",
                          "sum": "Importe",
                          "month": "Mes",
                      },
                      hover_data={
                          'categoria_name': True,
                          'sum': ':.2f',
                          'month': False
                      },
                      )
        bars.update_layout(hovermode="x unified")
        pie = px.pie(dfpies,
                     title="Gastos",
                     names='categoria_name',
                     color='categoria_name',
                     color_discrete_map=colors,
                     values='sum'
                     )
        bars.update_layout(hovermode="x")
        psums = []
        for index, row in dfscat.iterrows():
            if len(psums)>0:
                psums.append(psums[len(psums)-1]+row['sum'])
            else:
                psums.append(row['sum'])
        bars.add_trace(
            go.Scatter(
                x=dfscat['month'],
                y=psums,
                name="Neto",
                mode="lines",
                line=go.scatter.Line(color="black")
        ))
    else:
        bars = [html.Div('No data!')]
        pie = [html.Div('No data!')]

    return [
        dcc.Graph(
            id='balance_history',
            figure=bars,
        ),
        dcc.Graph(
            id='pies',
            figure=pie,
        )
    ]
