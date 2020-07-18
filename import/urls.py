
from django.urls import path
from django.views.generic.base import TemplateView
from .views import *

urlpatterns = [
    path('import/',  TemplateView.as_view(template_name='import/import.html'), name='import'),
    path('import_file', import_file, name='import_file'),
    path('parse_file', parse_file),
    path('categorize', categorize)
]