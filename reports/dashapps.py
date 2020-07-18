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


class HistorialBalance(object):

    def __init__(self, tickets, convert=False):

        app = DjangoDash('BalanceHistorial')
        graphs = []
        combos = []
        importe_field = 'importe'
        if convert:
            tickets = tickets.annotate(importe_conv=F('importe')*F('moneda__cambio'))
            importe_field='importe_conv'


        data = tickets.order_by('fecha')
        data = data.annotate(month=TruncMonth('fecha'))
        data1 = data.annotate(categoria_name=F('categoria__name')).values('month', 'categoria_name').order_by('month').annotate(sum=Sum(importe_field))
        df = pd.DataFrame(list(data1))
        if len(df) > 1:
            fig = px.bar(df, x='month', y='sum', color='categoria_name', labels={"categoria_name":"Categoria"})
        else:
            fig = [html.Div('No data!')]

        graphs.append(dcc.Graph(
            id='balance_history',
            figure=fig,
        ))

        options = []
        months = [i['month'].strftime('%m-%Y') for i in tickets.annotate(month=TruncMonth('fecha')).values('month').annotate(sum=Sum('importe'))]
        months = sorted(set(months), key=lambda x: datetime.datetime.strptime(x, '%m-%Y'), reverse=True)
        for month in months:
            options.append({
                'label': month,
                'value': month,
            })
        combos.append(dcc.Dropdown(id='month-selector',
                                   options=options,
                                   value=None,
                                   style=dict(width='40%')
                                   ))

        options = []
        years = [i['year'].strftime('%Y') for i in tickets.annotate(year=TruncYear('fecha')).values('year').annotate(sum=Sum('importe'))]
        years = sorted(set(years), key=lambda x: datetime.datetime.strptime(x, '%Y'), reverse=True)
        for year in years:
            options.append({
                'label': year,
                'value': year,
            })
        combos.append(dcc.Dropdown(id='year-selector',
                                   options=options,
                                   value=None,
                                   style=dict(width='40%')
                                   ))
        app.layout = html.Div([
            html.Div(id='result', children=None, style=dict(display='flex')),
            html.Div(id='comboarea', children=combos, style=dict(display='flex')),
            html.Div(id='plotarea', children=graphs),
        ])
        @app.callback(
        dash.dependencies.Output('result', 'children'),
        [dash.dependencies.Input('year-selector', 'value')],
        )
        def callback_year_select(year):
            children = []
            q = data.filter(fecha__year=year, importe__gte=0).order_by('categoria').values('categoria__name').annotate(sum=Sum(importe_field))
            children.append(
                html.Div(id='pie', children=
                dcc.Graph(id='in_pie', figure={'data': [
                go.Pie(
                    labels=[i['categoria__name'] for i in q],
                    values=[i['sum'] for i in q],
                    name='Income'
                )
              ]}))
            )
            q = data.filter(fecha__year=year, importe__lte=0).order_by('categoria').values('categoria__name').annotate(sum=Sum(importe_field))
            children.append(
                html.Div(id='pie2', children=
                dcc.Graph(id='out_pie', figure={'data': [
                go.Pie(
                    labels=[i['categoria__name'] for i in q],
                    values=[-i['sum'] for i in q],
                    name='Expenses'
                )
                ]}))
            )
            return children
