from django.urls import path
from .views import *


urlpatterns = [
    path('cuentas', cuentas, name='cuentas'),
    path('monedas', monedas, name='monedas'),
    path('modos_transferencia', modos_transferencia, name='modos_transferencia'),
    path('categorias', categorias, name='categorias'),
    path('proyectos', proyectos, name='proyectos'),
]