from django.urls import path
from django.views.generic.base import TemplateView
from .views import *
from nn_clasify.views import auto_classify

urlpatterns = [
    path('editar_ticket', edit_ticket),
    path('editar_categoria', edit_categoria),
    path('editar_proyecto', edit_proyecto),
    path('editar_cuenta', edit_cuenta),
    path('delete_ticket', delete_ticket),
    path('delete_categoria', delete_categoria),
    path('delete_proyecto', delete_proyecto),
    path('delete_cuenta', delete_cuenta),
    path('ticket', ticket),
    path('tickets', tickets),
]