from django.test import TestCase
from .nn import AutoClassifier
from contable.models import *
import datetime
import numpy as np
import torch.nn as nn
import torch
# Create your tests here.


class AutoClassifierTest(TestCase):
    def setUp(self):
        moneda = Moneda.objects.create(key="EUR", name="Euro", cambio=1)
        Categoria.objects.create(name="No Categorizado")
        self.categoria = Categoria.objects.create(name="Comida")
        proyecto = Proyecto.objects.create(name="No asignado")
        self.cuenta = Cuenta.objects.create(key="test", name="test", moneda=moneda)

        Ticket.objects.create(fecha=datetime.date.today(), concepto="Cafe",
                                            importe=100, cuenta=self.cuenta, moneda=moneda,
                                            categoria=self.categoria, proyecto=proyecto)
        self.ticket = Ticket.objects.create(fecha=datetime.date.today(), concepto="Cafe",
                              importe=100, cuenta=self.cuenta, moneda=moneda,
                              categoria=Categoria.objects.get(name="No Categorizado"), proyecto=proyecto)

        self.vectorized_ticket = torch.tensor([1, 1, 1, 1, 1, 1, 1], dtype=torch.float)

        self.classifier = AutoClassifier()

    def test_dictionary_traindata_setup(self):
        #prepare
        self.classifier.setup_train_data(self.cuenta)
        #assert
        self.assertEquals(self.classifier.dim_out, 2)

    def test_vectorize_ticket(self):
        #prepare
        self.classifier.setup_train_data(self.cuenta)
        expected_single_result = self.vectorized_ticket
        # act
        single_result = self.classifier.vectorize_ticket(self.ticket)
        #assert
        self.assertEqual(single_result.tolist(), expected_single_result.tolist())

    def test_vectorize_tickets(self):
        #prepare
        self.classifier.setup_train_data(self.cuenta)
        expected_results = torch.stack([self.vectorized_ticket, self.vectorized_ticket],0)

        #act
        results = self.classifier.vectorize_tickets(Ticket.objects.all())

        #assert
        self.assertEqual(len(results), len(expected_results))

        for i in range(len(results)):
            self.assertEqual(results[i].tolist(), expected_results[i].tolist())

    def test_vectorize_categoria(self):
        #prepare
        self.classifier.setup_train_data(self.cuenta)
        expected_results = np.zeros(len(self.classifier.out_dictionary))
        expected_results[self.classifier.out_dictionary.index(self.ticket.categoria.name)] = 1
        expected_results = torch.tensor([expected_results])

        #act
        results = self.classifier.vectorize_categoria(Ticket.objects.filter(id=self.ticket.id))

        #assert
        self.assertEqual(len(results), len(expected_results))
        for i in range(len(results)):
            self.assertEqual(results[i].tolist(), expected_results[i].tolist())

    def test_devectorize_categoria(self):
        #prepare
        self.classifier.setup_train_data(self.cuenta)
        input = np.zeros(len(self.classifier.out_dictionary))
        input[self.classifier.out_dictionary.index('Comida')] = 1
        input = torch.tensor(input)
        expected_result = Categoria.objects.get(name="Comida")

        #act
        result = self.classifier.devectorize_categoria(input)

        #assert
        self.assertEqual(result, expected_result)

    def test_simple_nn(self):
        #prepare
        self.classifier.setup_train_data(self.cuenta)
        training_tickets = Ticket.objects.filter(categoria=self.categoria)
        self.classifier.train_nn(training_tickets, iterations=1000, neurons=100)
        expected_result = Categoria.objects.get(name="Comida")

        #act
        result = self.classifier.predict(self.ticket)

        #assert
        self.assertIsInstance(self.classifier._module, nn.Sequential)
        self.assertEqual(result, expected_result)

    def test_save_load(self):
        # prepare
        self.classifier.setup_train_data(self.cuenta)
        training_tickets = Ticket.objects.filter(categoria=self.categoria)
        self.classifier.train_nn(training_tickets, iterations=1000, neurons=100)

        #act
        self.classifier.save_nn('torch_models/test-model')
        classifier2 = AutoClassifier()
        classifier2.load_nn('torch_models/test-model')

        #assert
        self.assertIsInstance(self.classifier._module, nn.Sequential)
        self.assertEqual(self.classifier.predict(self.ticket), classifier2.predict(self.ticket))
