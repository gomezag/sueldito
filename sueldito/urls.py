
from django.contrib import admin
from django.urls import path, include
from django.views.generic.base import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')),
    path('', TemplateView.as_view(template_name='home.html'), name='home'),
    path('import/',  include('import.urls')),
    path('view/',  include('reports.urls')),
    path('', include('contable.urls')),
    path('django_plotly_dash/', include('django_plotly_dash.urls')),
]