
from django.urls import path
from django.views.generic.base import TemplateView
from .views import *
from nn_clasify.views import auto_classify
urlpatterns = [
    path('import/', import_view, name='import'),
    path('import_file', import_file, name='import_file'),
    path('auto_classify', auto_classify , name='auto_classify'),
    path('parse_file', parse_file),
    path('categorize', categorize)
]