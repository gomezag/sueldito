from django.urls import path
from .views import *


urlpatterns = [
    path('cuenta/<int:cuenta_id>/', cuenta, name='cuenta_view'),
    path('categoria/<int:id>/', categoria, name='categoria_view'),

]