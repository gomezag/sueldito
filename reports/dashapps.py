import datetime

import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
from django_plotly_dash import DjangoDash

import contable.models as md

from plotly.subplots import make_subplots
from django.db.models import Sum, F
from django.db.models.functions import TruncMonth, TruncYear
from dash.dependencies import Input, Output, State

consulta = DjangoDash('Consulta')
cuenta_options = [dict(label=cuenta.name, value=cuenta.id) for cuenta in md.Cuenta.objects.all().order_by('name')]
cat_options = [dict(label=cat.name, value=cat.id) for cat in md.Categoria.objects.all().order_by('name')]
proy_options = [dict(label=cat.name, value=cat.id) for cat in md.Proyecto.objects.all().order_by('name')]

consulta.layout = html.Div([
    dcc.DatePickerRange(
        id='date-picker',
        start_date = (datetime.date.today()-datetime.timedelta(days=90)).replace(day=1),
        end_date = datetime.date.today(),
    ),
    dcc.Checklist(options=cuenta_options, id='cuenta-picker'),
    html.Div([dcc.Checklist(options=cat_options, id='cat-picker', value=[d['value'] for d in cat_options]),
              dcc.Checklist(
                  id="cat-all-or-none",
                  options=[{"label": "Select All", "value": "All"}],
                  value=[],
                  labelStyle={"display": "inline-block"},
              ),
              ]),
    html.Div([dcc.Checklist(options=proy_options, id='proy-picker', value=[d['value'] for d in proy_options]),
              dcc.Checklist(
                  id="proy-all-or-none",
                  options=[{"label": "Select All", "value": "All"}],
                  value=[],
                  labelStyle={"display": "inline-block"},
              ),
              ]),

    dcc.RadioItems(id='graph-type',
        options=[dict(label='Pie', value='pie'),
                             dict(label='Sankey', value='sankey')]),
    html.Div(id='plotarea')
])


@consulta.callback(
    Output("cat-picker", "value"),
    [Input("cat-all-or-none", "value")],
    [State("cat-picker", "options")],
)
def cat_select_all_none(all_selected, options, *args, **kwargs):
    all_or_none = []
    all_or_none = [option["value"] for option in options if all_selected]
    return all_or_none

@consulta.callback(
    Output("proy-picker", "value"),
    [Input("proy-all-or-none", "value")],
    [State("proy-picker", "options")],
)
def proy_select_all_none(all_selected, options, *args, **kwargs):
    all_or_none = []
    all_or_none = [option["value"] for option in options if all_selected]
    return all_or_none


@consulta.expanded_callback(
    Output('plotarea', 'children'),
    [
        Input('date-picker', 'start_date'),
        Input('date-picker', 'end_date'),
        Input('cuenta-picker', 'value'),
        Input('graph-type', 'value'),
        Input('cat-picker', 'value'),
        Input('proy-picker', 'value'),

    ]
)
def update_graph(start_date, end_date, sel_cuenta, graph_type, sel_cat, sel_proy, session_state=None, *args, **kwargs):
    if session_state is None:
        raise NotImplemented('Session not created')
    cuenta = session_state.get('cuenta', None)
    categoria = session_state.get('categoria', None)
    proyecto = session_state.get('proyecto', None)
    tickets = md.Ticket.objects.filter(fecha__gte=start_date,
                                       fecha__lte=end_date,
                                       categoria__in=md.Categoria.objects.all())

    if sel_cuenta:
        tickets = tickets.filter(cuenta__in=[md.Cuenta.objects.get(id=cuenta) for cuenta in sel_cuenta])
    if sel_cat:
        tickets = tickets.filter(categoria__in=[md.Categoria.objects.get(id=cat) for cat in sel_cat])
    if sel_proy:
        tickets = tickets.filter(proyecto__in=[md.Proyecto.objects.get(id=cat) for cat in sel_proy])
    cats = [cat for cat in md.Categoria.objects.all()]
    invest_tickets = tickets.filter(categoria__in=
                   md.Categoria.objects.filter(name__in=["Transferencia", "Inversiones"])
                   )
    inc_tickets = tickets.exclude(categoria__in=
                                  md.Categoria.objects.filter(name__in=["Transferencia", "Inversiones"])
                                  ).filter(importe__gt=0)
    inc_net = sum([t.importe*t.moneda.cambio for t in inc_tickets])
    exp_tickets = tickets.exclude(categoria__in=
                                  md.Categoria.objects.filter(name__in=["Transferencia", "Inversiones"])
                                  ).filter(importe__lt=0)
    exp_net = -sum([t.importe*t.moneda.cambio for t in exp_tickets])
    exp_df = pd.DataFrame(columns=['val', 'col'],
                          index=[cat.name for cat in cats],
                          data=[[-sum([t.importe * t.moneda.cambio for t in exp_tickets.filter(categoria=cat)]), cat.color] for cat in cats])
    inc_df = pd.DataFrame(columns=['val', 'col'],
                          index=[cat.name for cat in cats],
                          data=[[sum([t.importe*t.moneda.cambio for t in inc_tickets.filter(categoria=cat)]), cat.color] for cat in cats])

    sankey_bar_xpositions=[]
    sankey_bar_ypositions = []
    labels = []
    for cat in cats:
        labels.append("{}".format(cat.name))

        exp = -sum([t.importe*t.moneda.cambio for t in exp_tickets.filter(categoria=cat)])
        inc = sum([t.importe*t.moneda.cambio for t in inc_tickets.filter(categoria=cat)])

        if exp >= inc:
            sankey_bar_xpositions.append(0.8)
            sankey_bar_ypositions.append(0.5)
        else:
            sankey_bar_xpositions.append(0.2)
            sankey_bar_ypositions.append(0.5)

    sankey_bar_xpositions.append(0.4)
    sankey_bar_ypositions.append(0.5)
    labels.append("Income: {:,.2f}".format(inc_net))
    sankey_bar_xpositions.append(0.5)
    sankey_bar_ypositions.append(0.5)
    labels.append("Caja: {:,.2f}".format(inc_net-exp_net))
    sankey_bar_xpositions.append(0.6)
    sankey_bar_ypositions.append(0.5)
    labels.append("Expenses: {:,.2f}".format(exp_net))
    print(len(labels))
    print(len(sankey_bar_xpositions), len(sankey_bar_ypositions))
    graph = []

    fig = go.Figure(
        layout = go.Layout(
            title="Report",
            height=1400,
        )
    )
    fig.add_trace(go.Sankey(
            node=dict(
                line=dict(color="black", width=1),
                label=labels,
                customdata=["{:,.2f}".format(sum([t.importe*t.moneda.cambio for t in tickets.filter(categoria=c)])) for c in cats]+['1','2','3'],
                hovertemplate='%{customdata} Eur.',
                color=[*[c.color for c in cats],
                       'green',
                       'blue',
                       'red',
                       #*[c.color for c in cats]
                       ],

                pad=4,

            ),
            arrangement="snap",
            orientation='h',
            link=dict(
                source=[*[len(cats)+2 for t in exp_tickets], *[cats.index(t.categoria) for t in inc_tickets]
                    ,  len(cats), len(cats)+1
                        ],
                target=[*[cats.index(t.categoria) for t in exp_tickets], *[len(cats) for t in inc_tickets]
                    ,  len(cats)+1, len(cats)+2
                        ],
                value=[*[-t.importe*t.moneda.cambio for t in exp_tickets], *[t.importe*t.moneda.cambio for t in inc_tickets],  inc_net, exp_net],
                customdata=[*["{}: {:,.2f} {}".format(t.concepto, t.importe, t.moneda) for t in inc_tickets],
                            *["{}: {:,.2f} {}".format(t.concepto, t.importe, t.moneda) for t in exp_tickets],
                ],
                hovertemplate='%{customdata}',
                line=dict(color="black", width=0.1)
            )
        ))
    div = html.Div(children=[dcc.Graph(figure=fig)], style=dict(width="100%"))
    graph.append(div)

    fig = go.Figure(
        layout=go.Layout(
            height=500,
            width=500,
        )
    )
    labels = [i for i in inc_df.index]
    values = [val for val in inc_df['val']]

    fig.add_trace(go.Pie(labels=labels,
                         values=values,
                         text=["{}: {:,.0f}".format(labels[values.index(val)], val) if val / (sum(values)+0.01) > 0.05 else None
                               for val in values],
                         textinfo="text",
                         marker=dict(colors=inc_df['col'])
                         ))
    graph.append(dcc.Graph(figure=fig))

    fig = go.Figure(
        layout=go.Layout(
            height=500,
            width=500,
        )
    )
    labels = [i for i in exp_df.index]
    labels.append('Ahorro')
    values = [val for val in exp_df['val']]
    values.append(inc_net-exp_net)
    cols = [col for col in exp_df['col']]
    cols.append('green')
    fig.add_trace(go.Pie(labels=labels,
                         values=values,
                         text=["{}: {:,.0f}".format(labels[values.index(val)], val) if val/(sum(values)+0.01) > 0.05 else None for val in values],
                         textinfo="text",
                         marker=dict(colors=cols)
                         )
                  )
    div = html.Div(children=[dcc.Graph(figure=fig)], style=dict(width="100%"))
    graph.append(div)

    return graph

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
    dcc.RadioItems(
        id='grouping',
        options=[
            {'label': 'Categorias', 'value': 'categoria'},
            {'label': 'Proyectos', 'value': 'proyecto'}
        ],
        value='categoria',
        labelStyle={'display': 'inline-block'}
    ),
    html.Div(id='plotarea', children=graphs),
])

@historial.expanded_callback(
    Output('plotarea', 'children'),
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('convert', 'value'),
     Input('grouping', 'value')]
)
def update_graph(start_date, end_date, convert, group, session_state=None, *args, **kwargs):
    import contable.models as md
    if session_state is None:
        raise NotImplemented('Session not created')
    cuenta = session_state.get('cuenta', None)
    categoria= session_state.get('categoria', None)
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

    if group == 'categoria':
        tickets = tickets.annotate(categoria_name=F('categoria__name'))
    elif group == 'proyecto':
        tickets = tickets.annotate(categoria_name=F('proyecto__name'))

    tickets = tickets.annotate(month=TruncMonth('fecha'))

    dfbars = pd.DataFrame(list(tickets.values('month', 'categoria_name').order_by(
        'month').annotate(sum=Sum('importe_conv'))))

    dfscat = pd.DataFrame(list(tickets.values('month').order_by(
        'month').annotate(sum=Sum('importe_conv'))))

    dfpies = []

    for title in ['ingresos', 'gastos']:
        if title == 'ingresos':
            tipie = tickets.values('categoria_name').annotate(sum=Sum('importe_conv'))
            pie = pd.DataFrame(list(tipie))
        else:
            tipie = tickets.values('categoria_name').annotate(sum=-Sum('importe_conv'))
            pie = pd.DataFrame(list(tipie))
        savings = sum([ticket.importe_conv for ticket in tickets])

        if savings > 0 and title=='gastos':
            pie=pie.append(pd.DataFrame(columns=pie.columns, data=[('Savings', savings)]))
        print(pie)
        print(savings)
        dfpies.append((title,pie))

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
        pies = [px.pie(pie,
                     title=title,
                     names='categoria_name',
                     color='categoria_name',
                     color_discrete_map=colors,
                     values='sum'
                     ) for title,pie in dfpies]
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
        pies = [html.Div('No data!')]
    fig = [dcc.Graph(
        id='balance_history',
        figure=bars)]
    i=0
    for pie in pies:
        i+=1
        fig.append(
            dcc.Graph(
                id='pies'+str(i),
                figure=pie,
            )
        )

    return fig

