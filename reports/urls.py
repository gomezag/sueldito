from django.urls import path
from .views import *
from api.views import tickets

urlpatterns = [
    path('cuenta/<int:cuenta_id>/', cuenta, name='cuenta_view'),
    path('cuenta/<int:cuenta_id>/tickets', tickets, name='cuenta_tickets'),
    path('categoria/<int:categoria_id>/', categoria, name='categoria_view'),
    path('categoria/<int:categoria_id>/tickets', tickets, name='categoria_tickets'),
    path('proyecto/<int:proyecto_id>/', proyecto, name='proyecto_view'),
    path('proyecto/<int:proyecto_id>/tickets', tickets, name='proyecto_tickets'),
#    path('categorizar', categorize),
    path('consulta',consulta, name="sankey"),
]