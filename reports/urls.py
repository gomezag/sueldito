from django.urls import path
from .views import *


urlpatterns = [
    path('cuenta/<int:cuenta_id>/', cuenta, name='cuenta_view'),
    path('cuenta/<int:cuenta_id>/tickets', get_tickets, name='cuenta_tickets'),
    path('categoria/<int:categoria_id>/', categoria, name='categoria_view'),
    path('categoria/<int:categoria_id>/tickets', get_tickets, name='categoria_tickets'),
    path('categorizar', categorize),
    path('editar_ticket', edit_ticket),
    path('editar_categoria', edit_categoria),
    path('delete_categoria', delete_categoria),
    path('delete_proyecto', delete_proyecto),
    path('borrar', delete_ticket),
    path('proyecto/<int:proyecto_id>/', proyecto, name='proyecto_view'),
    path('proyecto/<int:proyecto_id>/tickets', get_tickets, name='proyecto_tickets'),
]