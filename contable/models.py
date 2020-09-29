from django.db import models
from django.db.models import Sum
from django.db.models.signals import post_save
from django.utils.functional import cached_property
from colorfield.fields import ColorField
# Create your models here.

class Categoria(models.Model):
    name = models.CharField(max_length=200)
    color = ColorField(default='#FF0000', null=True, blank=True)

    def __str__(self):
        return self.name


class Proyecto(models.Model):
    name = models.CharField(max_length=200)
    color = models.CharField(max_length=10, null=True, blank=True)
    def __str__(self):
        return self.name


class ModoTransferencia(models.Model):
    name = models.CharField(max_length=15)
    key = models.CharField(max_length=2)

    def __str__(self):
        return self.name


class Moneda(models.Model):
    name = models.CharField(max_length=20)
    key = models.CharField(max_length=3)
    cambio = models.FloatField(default=1)

    def __str__(self):
        return self.name


class Cuenta(models.Model):
    name = models.CharField(max_length=200)
    key = models.CharField(max_length=16, default='NONE')
    moneda = models.ForeignKey(Moneda, on_delete=models.PROTECT)

    def __str__(self):
        return self.name

    @property
    def balance(self):
        tickets = self.ticket_set.all()
        importe = sum([ti.moneda.cambio * ti.importe for ti in tickets])
        return importe

    def last_update_date(self):
        tickets = self.ticket_set.all()
        min_date = max([ti.fecha for ti in tickets])
        return min_date


def get_uncategorized():
    return Categoria.objects.get(name="No Categorizado")


def get_unassigned():
    return Proyecto.objects.get(name="No asignado")

class Ticket(models.Model):
    fecha = models.DateField('Fecha')
    cuenta = models.ForeignKey(Cuenta, on_delete=models.CASCADE)
    importe = models.FloatField('importe')
    moneda = models.ForeignKey(Moneda, on_delete=models.SET_NULL, null=True, blank=True)
    modo = models.ForeignKey(ModoTransferencia, on_delete=models.SET_NULL, null=True, blank=True)
    concepto = models.CharField(max_length=200)
    categoria = models.ForeignKey(Categoria, on_delete=models.SET(get_uncategorized),
                                  null=False, blank=False)
    consistency = models.BooleanField(default=False)
    proyecto = models.ForeignKey(Proyecto, on_delete=models.SET(get_unassigned),
                                 null=False, blank=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        string = str(self.fecha) + ' ' + str(self.concepto)
        return string

    def importe_consistency_check(self):
        if float(self.subticket_set.all().aggregate(Sum('importe'))['importe__sum']) == self.importe:
            self.consistency = True
            self.save()
            return True
        else:
            self.consistency = False
            self.save()
            return False

    def categorize(self):
        for cat in Categoria.objects.all():
            for tag in cat.tags:
                if tag in self.concepto:
                    self.categoria = cat

        if self.categoria is None:
            self.categoria = Categoria.objects.get(key='NC')
        self.save()

    @cached_property
    def saldo(self):
        tickets = Ticket.objects.filter(fecha__lte=self.fecha, cuenta=self.cuenta)
        return sum([ticket.importe for ticket in tickets])


class SubTicket(models.Model):
    ticket = models.ForeignKey(Ticket, on_delete=models.CASCADE)
    concepto = models.CharField(max_length=200)
    importe = models.FloatField('importe')
    categoria = models.CharField(max_length=200, default="Total")
    transaccion = models.ForeignKey(ModoTransferencia, on_delete=models.SET_NULL, null=True)

    def __str__(self):
        return self.ticket.__str__() + ': ' + self.concepto


def create_subticket_on_ticket_creation(sender, instance, created, **kwargs):
    if instance and created:
        t = Ticket.objects.get(pk=instance.id)
        t.subticket_set.create(importe=instance.importe, concepto="N.C.",
                               transaccion=t.modo)


def check_ticket_consistency_after_subticket_creation(sender, instance, created, **kwargs):
    if instance and created:
        t = Ticket.objects.get(pk=instance.ticket.id)
        return t.importe_consistency_check()


post_save.connect(check_ticket_consistency_after_subticket_creation, sender=SubTicket)
post_save.connect(create_subticket_on_ticket_creation, sender=Ticket)
