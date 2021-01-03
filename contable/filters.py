import django_filters as df
from django_filters.widgets import RangeWidget

from .models import *

class TicketFilter(df.FilterSet):
    fecha = df.DateFromToRangeFilter(widget=RangeWidget(attrs={'type': 'date'}))
    importe = df.RangeFilter()

    class Meta:
        model = Ticket
        fields = ['fecha', 'importe', 'concepto', 'categoria', 'cuenta', ]