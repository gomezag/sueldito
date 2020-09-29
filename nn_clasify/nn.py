from django.db.models.functions import ExtractWeekDay, ExtractMonth, ExtractDay
from django.db.models import F
from contable.models import *
from torch.optim import Adam
import torch

import numpy as np
import torch.nn as nn


class AutoClassifier(nn.Module):
    def __init__(self):
        super(AutoClassifier, self).__init__()
        self.in_dictionary = dict()
        self.out_dictionary = []
        self.model_in_keys = ['moy', 'dow', 'dom', 'importe_conv', 'mon', 'cue', 'con']
        self.model_out_key = 'cat'
        self.dim_in = 0
        self.dim_out = 0
        self._module = None

    def annotate_tickets(self, tickets):
        tickets = tickets.annotate(dow=ExtractWeekDay('fecha'),
                                   moy=ExtractMonth('fecha'),
                                   dom=ExtractDay('fecha'),
                                   mon=F('moneda__key'),
                                   cue=F('cuenta__name'),
                                   cat=F('categoria__name'),
                                   con=F('concepto'),
                                   importe_conv=F('importe') * F('moneda__cambio'))
        tickets = tickets.values(*np.concatenate([self.model_in_keys, [self.model_out_key]]))
        return tickets

    def annotate_ticket(self, ticket):
        result = dict(
            dow=(ticket.fecha.isoweekday() % 7) + 1,
            moy=ticket.fecha.month,
            dom=ticket.fecha.day,
            mon=ticket.moneda.key,
            cue=ticket.cuenta.name,
            cat=ticket.categoria.name,
            con=ticket.concepto,
            importe_conv=ticket.importe*ticket.moneda.cambio,
        )
        return result

    def setup_train_data(self,cuenta):
        tickets = Ticket.objects.all().filter(cuenta=cuenta)
        tickets = self.annotate_tickets(tickets)

        self.dim_in = 0
        for key in self.model_in_keys:
            self.in_dictionary[key] = list(set([ticket[key] for ticket in tickets.values(key)]))
            self.dim_in += len(self.in_dictionary[key])

        self.out_dictionary = list(set([ticket[self.model_out_key] for ticket in tickets.values(self.model_out_key)]))
        self.dim_out = len(self.out_dictionary)


    def vectorize_tickets(self, tickets):
        tickets = list(self.annotate_tickets(tickets))
        results = []
        for ticket in tickets:
            vector = np.array([])
            for key in self.model_in_keys:
                sub_vector = np.array([
                                0 if i != self.in_dictionary[key].index(ticket[key])
                                else 1
                                for i in range(len(self.in_dictionary[key]))
                             ])
                vector = np.concatenate([vector, sub_vector])
            results.append(vector)

        results = torch.tensor(results, dtype=torch.float)
        return results

    def vectorize_ticket(self, ticket):
        ticket = self.annotate_ticket(ticket)
        results = []

        vector = np.array([])
        for key in self.model_in_keys:
            sub_vector = np.array([
                            0 if i != self.in_dictionary[key].index(ticket[key])
                            else 1
                            for i in range(len(self.in_dictionary[key]))
                         ])
            vector = np.concatenate([vector, sub_vector])
        results = vector

        results = torch.tensor(results, dtype=torch.float)
        return results

    def vectorize_categoria(self, tickets):
        tickets = self.annotate_tickets(tickets)
        key = self.model_out_key

        results = []

        for ticket in tickets:
            results.append(np.array([
                0 if i != self.out_dictionary.index(ticket[key])
                else 1
                for i in range(len(self.out_dictionary))
            ]))
        results = torch.tensor(results, dtype=torch.float)
        return results

    def devectorize_categoria(self, cat_vector):
        index_of_max = torch.argmax(cat_vector).item()
        if cat_vector[index_of_max] > 0.7:
            cat = self.out_dictionary[index_of_max]
            cat = Categoria.objects.get(name=cat)
            return cat
        else:
            print("no prediction at "+str(cat_vector[index_of_max]))
            return None

    def train_nn(self, train_tickets, iterations=1000, neurons=10):
        device = torch.device("cpu")
        self.dim_h = neurons

        self._module = nn.Sequential(nn.Linear(self.dim_in, self.dim_h),
            nn.ReLU(),
            nn.Linear(self.dim_h, self.dim_out),
            nn.Sigmoid(),
            )

        loss_fn = nn.MSELoss()
        learning_rate = 1e-4
        optimizer = Adam(self._module.parameters(), lr=learning_rate, weight_decay=0.2)
        drop = nn.Dropout(p=0.3)
        X = self.vectorize_tickets(train_tickets)
        Y = self.vectorize_categoria(train_tickets)

        for t in range(iterations):
            y_pred = self._module(drop(X))

            loss = loss_fn(y_pred, Y)
            if t % 100 == 99:
                print(t, loss.item())
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

    def predict(self, ticket):
        x = self.vectorize_ticket(ticket)
        y = self._module(x)

        return self.devectorize_categoria(y)