from django.urls import path
from .views import *


urlpatterns = [
    path('cuenta/<int:cuenta_id>/', cuenta, name='cuenta_view'),
    path('cuenta/<int:cuenta_id>/tickets', get_tickets, name='cuenta_tickets'),
    path('categoria/<int:categoria_id>/', categoria, name='categoria_view'),
    path('categoria/<int:categoria_id>/tickets', get_tickets, name='categoria_tickets'),
    path('proyecto/<int:proyecto_id>/', proyecto, name='proyecto_view'),
    path('proyecto/<int:proyecto_id>/tickets', get_tickets, name='proyecto_tickets'),
#    path('categorizar', categorize),
    path('consulta',consulta, name="sankey"),
]