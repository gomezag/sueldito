from django.db.models.functions import ExtractWeekDay, ExtractMonth, ExtractDay
from django.db.models import F
from contable.models import *
import re
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.modules import Module
from torch import Tensor
import numpy as np

import pickle as pkl


class maxs(Module):
    def forward(self, input:Tensor) -> Tensor:
        output = torch.tensor(
            [
                [1 if i == max(row) else 0 for i in row]
                for row in input.tolist()]
        )

        return output




class AutoClassifier(nn.Module):
    def __init__(self):
        super(AutoClassifier, self).__init__()
        self.in_dictionary = dict()
        self.out_dictionary = []
        self.model_in_keys = ['dow','dom','con', 'importe_conv']
        self.model_out_key = 'cat'
        self.dim_in = 0
        self.dim_out = 0
        self.dim_h = 10
        self._module = None
        self.categoria = None

    def annotate_tickets(self, tickets, io):

        return [self.annotate_ticket(ticket, io=io) for ticket in tickets]

    def annotate_ticket(self, ticket, io):
        if io == 'in':
            result = dict(
                dow=(ticket.fecha.isoweekday() % 7) + 1,
                moy=ticket.fecha.month,
                dom=ticket.fecha.day,
                mon=ticket.moneda.key,
                cue=ticket.cuenta.name,
                con=[''.join(e for e in re.sub("\d+","",con) if e.isalnum()) for con in ticket.concepto.split(' ')],
                importe_conv=ticket.importe*ticket.moneda.cambio,
                fecha=ticket.fecha,
            )
            return result

        elif io == 'out':
            result = dict(
                cat=ticket.categoria.name,
            )
            return result

    def setup_train_data(self):
        tickets = Ticket.objects.all()
        tickets_in = self.annotate_tickets(tickets, io='in')
        self.dim_in = 0
        for key in self.model_in_keys:
            if key == "importe_conv":
                self.dim_in += 1
            elif key == "con":
                dictionary = []
                for ticket in tickets_in:
                    dictionary += ticket[key]
                self.in_dictionary[key] = sorted(list(set(dictionary)))
                self.dim_in += len(self.in_dictionary[key])
            else:
                self.in_dictionary[key] = list(set([ticket[key] for ticket in tickets_in]))
                self.dim_in += len(self.in_dictionary[key])

        tickets_out = self.annotate_tickets(tickets, io='out')
        self.out_dictionary = list(set([ticket[self.model_out_key] for ticket in tickets_out]))
        self.dim_out = len(self.out_dictionary)

    def vectorize_tickets(self, tickets, max_importe=1):
        results = torch.stack([self.vectorize_ticket(ticket, max_importe) for ticket in tickets], 0)

        return results

    def vectorize_ticket(self, ticket, max_importe=1):
        ticket = self.annotate_ticket(ticket, io='in')

        vector = []
        for key in self.model_in_keys:
            if key == "importe_conv":
                sub_vector = [ticket["importe_conv"] / max_importe]
            elif key == "con":
                index = [self.in_dictionary[key].index(i) for i in ticket[key]]
                sub_vector = [0.5 if i in index else -0.5 for i in range(len(self.in_dictionary[key]))]
            else:
                sub_vector = [
                    -0.5 if i != self.in_dictionary[key].index(ticket[key])
                    else 0.5
                    for i in range(len(self.in_dictionary[key]))
                ]
            vector += sub_vector

        result = torch.tensor(vector, dtype=torch.float)
        return result

    def vectorize_categoria(self, tickets):
        key = self.model_out_key

        tickets = self.annotate_tickets(tickets, io='out')
        results = []

        for ticket in tickets:
            results.append(np.array([
                0 if i != self.out_dictionary.index(ticket[key])
                else 1
                for i in range(len(self.out_dictionary))
            ]))

        results = torch.tensor(results, dtype=torch.float)
        return results

    def devectorize_categoria(self, cat_vector, p):
        index_of_max = torch.argmax(cat_vector).item()
        try:
            if torch.max(cat_vector)>p:
                cat = self.out_dictionary[index_of_max]
                cat = Categoria.objects.get(name=cat)
                return cat
            else:
                return Categoria.objects.get(name="No Categorizado")
        except Exception as e:
            repr(e)
            #print("no prediction at "+str(cat_vector[index_of_max]))
            return None

    def _init_nn(self):
        self._module = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_h),
            nn.ReLU(),
            # nn.Linear(self.dim_h, self.dim_h),
            # nn.Tanh(),
            nn.Linear(self.dim_h, self.dim_out),
            nn.Softmax(),
            )

    def train_nn(self, train_tickets, iterations=1000, neurons=10, reset_nn=False, lr=1e-4):
        device = torch.device("cpu")
        if reset_nn or self._module is None:
            self.dim_h = neurons
            self._init_nn()

        loss_fn = nn.BCELoss()
        learning_rate = lr
        optimizer = torch.optim.SGD(self._module.parameters(), lr=learning_rate, momentum=0.9)
        #drop = nn.Dropout(p=0.3)
        max_importe = max([np.abs(ticket.importe) for ticket in train_tickets])
        X = self.vectorize_tickets(train_tickets, max_importe=max_importe)
        Y = self.vectorize_categoria(train_tickets)

        for t in range(iterations):
            y_pred = self._module(X)

            loss = loss_fn(y_pred, Y)
            if t % 100 == 99:
                print(t, loss.item())
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

    def predict(self, ticket, p):
        x = self.vectorize_ticket(ticket).t()
        y = self._module(x)

        return self.devectorize_categoria(y, p)

    def save_nn(self, filename):
        torch.save(self._module.state_dict(), filename+'.torch')
        conf = dict(
            in_dictionary=self.in_dictionary,
            out_dictionary=self.out_dictionary,
            model_in_keys=self.model_in_keys,
            model_out_keys=self.model_out_key,
            dim_in=self.dim_in,
            dim_out=self.dim_out,
            dim_h=self.dim_h,
        )
        with open(filename+'.conf', 'wb') as f:
            pkl.dump(conf,f)

    def load_nn(self, filename):
        with open(filename+'.conf', 'rb') as f:
            conf = pkl.load(f)

        for key in ['in_dictionary', 'out_dictionary', 'model_in_keys', 'model_out_keys', 'dim_in', 'dim_out', 'dim_h']:
            self.__setattr__(key, conf[key])

        self._init_nn()
        self._module.load_state_dict(torch.load(filename+'.torch'))
        self._module.eval()

    def test_dataset(self, tickets, p):
        hits = 0
        miss = 0
        nc = 0
        for ticket in tickets:
            if self.predict(ticket, p).name == "No Categorizado":
                nc += 1
            elif self.predict(ticket, p).id == ticket.categoria.id:
                hits+=1
            else:
                miss += 1
        hits /= len(tickets)
        miss /= len(tickets)
        nc /= len(tickets)
        return hits, miss, nc


class maxs(Module):
    def forward(self, input:Tensor) -> Tensor:
        output = torch.tensor(
            [
                [1 if i == max(row) else 0 for i in row]
                for row in input.tolist()]
        )

        return output




class CategoriaClassifier(nn.Module):
    def __init__(self):
        super(CategoriaClassifier, self).__init__()
        self.in_dictionary = dict()
        self.out_dictionary = []
        self.model_in_keys = ['con', 'importe_conv']
        self.model_out_key = 'cat'
        self.dim_in = 0
        self.dim_out = 0
        self.dim_h = 10
        self._module = None
        self.categoria = None
        self.layers = None

    def annotate_tickets(self, tickets, io):

        return [self.annotate_ticket(ticket, io=io) for ticket in tickets]

    def annotate_ticket(self, ticket, io):
        if io == 'in':
            result = dict(
                dow=(ticket.fecha.isoweekday() % 7) + 1,
                moy=ticket.fecha.month,
                dom=ticket.fecha.day,
                mon=ticket.moneda.key,
                cue=ticket.cuenta.name,
                con=[''.join(e for e in re.sub("\d+","",con) if e.isalnum()) for con in ticket.concepto.split(' ')],
                importe_conv=ticket.importe*ticket.moneda.cambio,
                fecha=ticket.fecha,
            )
            return result

        elif io == 'out':
            result = dict(
                cat=ticket.categoria.name,
            )
            return result

    def setup_train_data(self, categoria):
        tickets = Ticket.objects.all()
        tickets_in = self.annotate_tickets(tickets, io='in')
        self.categoria = categoria
        self.dim_in = 0
        for key in self.model_in_keys:
            if key == "importe_conv":
                self.dim_in += 1
            elif key == "con":
                dictionary = []
                for ticket in tickets_in:
                    dictionary += ticket[key]
                self.in_dictionary[key] = sorted(list(set(dictionary)))
                self.dim_in += len(self.in_dictionary[key])
            else:
                self.in_dictionary[key] = list(set([ticket[key] for ticket in tickets_in]))
                self.dim_in += len(self.in_dictionary[key])

        self.dim_out = 1

    def vectorize_tickets(self, tickets, max_importe=1):
        results = torch.stack([self.vectorize_ticket(ticket, max_importe) for ticket in tickets], 0)

        return results

    def vectorize_ticket(self, ticket, max_importe=1):
        ticket = self.annotate_ticket(ticket, io='in')

        vector = []
        for key in self.model_in_keys:
            if key == "importe_conv":
                sub_vector = [ticket["importe_conv"] / max_importe]
            elif key == "con":
                index = [self.in_dictionary[key].index(i) for i in ticket[key]]
                sub_vector = [0.5 if i in index else -0.5 for i in range(len(self.in_dictionary[key]))]
            else:
                sub_vector = [
                    -0.5 if i != self.in_dictionary[key].index(ticket[key])
                    else 0.5
                    for i in range(len(self.in_dictionary[key]))
                ]
            vector += sub_vector

        result = torch.tensor(vector, dtype=torch.float)
        return result

    def vectorize_categoria(self, tickets):
        key = self.model_out_key
        categoria = self.categoria

        tickets = self.annotate_tickets(tickets, io='out')
        results = []
        for ticket in tickets:
            if ticket[key] == categoria.name:
                results.append([1])
            else:
                results.append([0])
        results = torch.tensor(results, dtype=torch.float)
        return results

    def devectorize_categoria(self, cat_vector, p):
        if cat_vector == [1]:
            return self.categoria
        else:
            return Categoria.objects.get(name="No Categorizado")

    def _init_nn(self, layers):
        funcs = [
            nn.Linear(self.dim_in, self.dim_h),
            nn.ReLU()]
        for i in range(layers-1):
            funcs.append(nn.Linear(self.dim_h, self.dim_h))
            funcs.append(nn.ReLU())
        funcs.append(nn.Linear(self.dim_h, self.dim_out))
        funcs.append(nn.Sigmoid())
        self._module = nn.Sequential(*funcs)

    def train_nn(self, train_tickets, iterations=1000, neurons=10, reset_nn=False, lr=1e-4, layers=1):
        device = torch.device("cpu")
        if reset_nn or self._module is None:
            self.dim_h = neurons
            self._init_nn(layers)

        loss_fn = nn.BCELoss()
        learning_rate = lr
        optimizer = torch.optim.Adam(self._module.parameters(), lr=learning_rate, weight_decay=0.9)
        #drop = nn.Dropout(p=0.3)
        max_importe = max([np.abs(ticket.importe) for ticket in train_tickets])
        X = self.vectorize_tickets(train_tickets, max_importe=max_importe)
        Y = self.vectorize_categoria(train_tickets)

        for t in range(iterations):
            y_pred = self._module(X)

            loss = loss_fn(y_pred, Y)
            if t % 100 == 99:
                print(t, loss.item())
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

    def predict(self, ticket, p):
        x = self.vectorize_ticket(ticket).t()
        y = self._module(x)

        return self.devectorize_categoria(y, p)

    def save_nn(self, filename):
        torch.save(self._module.state_dict(), filename+'.torch')
        conf = dict(
            in_dictionary=self.in_dictionary,
            out_dictionary=self.out_dictionary,
            model_in_keys=self.model_in_keys,
            model_out_keys=self.model_out_key,
            dim_in=self.dim_in,
            dim_out=self.dim_out,
            dim_h=self.dim_h,
        )
        with open(filename+'.conf', 'wb') as f:
            pkl.dump(conf,f)

    def load_nn(self, filename):
        with open(filename+'.conf', 'rb') as f:
            conf = pkl.load(f)

        for key in ['in_dictionary', 'out_dictionary', 'model_in_keys', 'model_out_keys', 'dim_in', 'dim_out', 'dim_h']:
            self.__setattr__(key, conf[key])

        self._init_nn()
        self._module.load_state_dict(torch.load(filename+'.torch'))
        self._module.eval()

    def test_dataset(self, tickets, p):
        hits = 0
        miss = 0
        nc = 0
        for ticket in tickets:
            if self.predict(ticket, p).name == "No Categorizado":
                nc += 1
            elif self.predict(ticket, p).id == ticket.categoria.id:
                hits+=1
            else:
                miss += 1
        hits /= len(tickets)
        miss /= len(tickets)
        nc /= len(tickets)
        return hits, miss, nc

