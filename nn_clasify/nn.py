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
import pandas as pd

import pickle as pkl

"""
Neural networks organized in classes with methods to translate data from the models into vectors and vice-versa.
The purpose is to automatically classify an incoming ticket using the following fields:
    - concepto
    - fecha
    - importe
    - moneda
    - cuenta

Other fields do not necessarily come with the import file, so are ignored. 
"""


class maxs(Module):
    def forward(self, input: Tensor) -> Tensor:
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
        self.model_in_keys = ['con', 'importe_conv', 'dow', 'dom', 'moy']
        self.model_in_keys = ['con']

        self.model_out_key = 'cat'
        self.dim_in = 0
        self.dim_out = 0
        self.dim_h = 10
        self._model = None
        self.categoria = None
        self.loss_fn = None

    def annotate_tickets(self, tickets, io):

        return [self.annotate_ticket(ticket, io=io) for ticket in tickets]

    def annotate_ticket(self, ticket, io):
        """
        Serialization of a ticket.
        TODO: move this to serializer class.

        :param ticket: ticket to vectorize
        :param io: whether we want to annotate the input fields or the output fields. Useful when we want to change the
         behaviour
        :return:
        """
        if io == 'in':
            result = dict(
                dow=(ticket.fecha.isoweekday() % 7) + 1,
                moy=ticket.fecha.month,
                dom=ticket.fecha.day,
                mon=ticket.moneda.key,
                cue=ticket.cuenta.name,
                con=[''.join(e for e in re.sub("\d+", "", con) if e.isalnum()) for con in ticket.concepto.split(' ')],
                importe_conv=ticket.importe * ticket.moneda.cambio,
                fecha=ticket.fecha,
            )
            return result

        elif io == 'out':
            result = dict(
                cat=ticket.categoria.name,
            )
            return result

    def setup_train_data(self, train):
        """
        create lists that will define the way a ticket is vectorized out of all the tickets stored in the
        database. Each field is converted in its own way and the dimension is calculated.
            - importe_conv: the number will be passed as is, no need for a dictionary so dim_in += 1
            - concepto: the body of text is converted to an array of 1s and 0s for each word.
                        It adds to the in_dictionary field an ordered list of words.
                        TODO: clean the words.
                        TODO: split the words.
                        TODO: add Google Maps data to the vector.

        :return: dim_in, dim_out
        """
        tickets = train
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
                self.in_dictionary[key] = sorted(list(set([ticket[key] for ticket in tickets_in])))
                self.dim_in += len(self.in_dictionary[key])

        tickets_out = self.annotate_tickets(tickets, io='out')
        self.out_dictionary = sorted(list(set([ticket[self.model_out_key] for ticket in tickets_out])))
        self.dim_out = len(self.out_dictionary)
        #self.dim_out = 1
        return self.dim_in, self.dim_out

    def vectorize_tickets(self, tickets, max_importe=1):
        """

        :param tickets: contable.models.Ticket QuerySet
        :param max_importe: scale factor of the importe value of a ticket. For normalization.
        :return: an input tensor.
        """
        results = torch.stack([self.vectorize_ticket(ticket, max_importe) for ticket in tickets], 0)

        return results

    def vectorize_ticket(self, ticket, max_importe=1):
        """
        Create a vector from a contable.models.Ticket object or QuerySet
        :param ticket: contable.models.Ticket object
        :param max_importe: normalization factor for the value of the ticket.
        :return: a vector representing the ticket.
        """
        ticket = self.annotate_ticket(ticket, io='in')

        vector = []
        for key in self.model_in_keys:
            l = len(self.in_dictionary[key])
            if key == "importe_conv":
                sub_vector = [ticket["importe_conv"] / max_importe]

            elif key == "con":
                try:
                    indexes = [self.in_dictionary[key].index(i) for i in ticket[key]]
                    sub_vector = [0.5 if i in indexes else -0.5 for i in range(l)]
                except ValueError:
                    sub_vector = [-0.5 for i in range(l)]
            else:
                try:
                    sub_vector = [
                        -0.5 if i != self.in_dictionary[key].index(ticket[key])
                        else 0.5
                        for i in range(l)
                    ]
                except ValueError:
                    sub_vector = [-0.5 for i in range(len(self.in_dictionary[key]))]

            vector += sub_vector

        result = torch.tensor(vector, dtype=torch.float)

        return result

    def vectorize_categoria(self, tickets):
        """
        Create an output ticket from a categorized ticket. Useful for creating training data.
        :param tickets: contable.models.Ticket object
        :return: a vector representing the ticket
        """
        key = self.model_out_key

        tickets = self.annotate_tickets(tickets, io='out')
        results = []

        for ticket in tickets:
            results.append(np.array([
                -10 if i != self.out_dictionary.index(ticket[key])
                else 0
                for i in range(len(self.out_dictionary))
            ]))

        results = torch.tensor(results, dtype=torch.float)
        return results

    def devectorize_categoria(self, cat_vector, p):
        """
        Inverse of vectorize_categoria.
        In this case, it will select the greatest value in the vector and set the category to this.
        :param cat_vector: vector with different values for each category
        :param p: set a minimum threshold for categorizing.
        :return: a contable.models.Category object.
        """
        index_of_max = torch.argmax(cat_vector).item()
        try:
            if torch.max(cat_vector).item() > -p:
                cat = self.out_dictionary[index_of_max]
                cat = Categoria.objects.get(name=cat)
                return cat
            else:
                return Categoria.objects.get(name="No Categorizado")
        except Exception as e:
            repr(e)
            # print("no prediction at "+str(cat_vector[index_of_max]))
            return None

    def _init_nn(self, layers):
        """
        Initialize the Neural Network instance with the dimensions stored.
        :return:
        """
        self._model = Classifier(self.dim_in, self.dim_out, self.dim_h, layers=layers)
        self.layers = layers
        # funcs = [
        #     nn.Linear(self.dim_in, self.dim_h),
        #     nn.ReLU()]
        # for i in range(layers - 1):
        #     funcs.append(nn.Linear(self.dim_h, self.dim_h))
        #     funcs.append(nn.ReLU())
        # funcs.append(nn.Linear(self.dim_h, self.dim_out))
        # # funcs.append(nn.Sigmoid())
        # funcs.append(nn.Softmax(dim=0))
        # self._module = nn.Sequential(*funcs)

    def train_nn(self, train_tickets, dev_tickets, iterations=1000,
                 neurons=40, layers=4, reset_nn=False, lr=1e-4, optim="SGD",
                 dropout=True):
        """
        Train the neural network.
        :param train_tickets:
        :param iterations:
        :param neurons: this is only useful if you're resetting the model.
        :param reset_nn: reset the model. If False, it will train over the stored model.
        :param lr: learning rate
        :param optim: optimizer. Options are Adam and SGD
        :return: None
        """
        device = torch.device("cpu")
        loss_evo = pd.DataFrame(columns=["train", "dev"], index=range(iterations))
        if reset_nn or self._model is None:
            self.dim_h = neurons
            self._init_nn(layers)
        if dropout:
            self._model.training = True
        #m = nn.LogSoftmax(dim=1)

        learning_rate = lr
        if optim == "SGD":
            optimizer = torch.optim.SGD(self._model.parameters(), lr=learning_rate, momentum=0.9)
        elif optim == "Adam":
            optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate, weight_decay=0.9)
        max_importe = max([np.abs(ticket.importe) for ticket in train_tickets])
        X_train = self.vectorize_tickets(train_tickets, max_importe=max_importe)
        Y_train = self.vectorize_categoria(train_tickets)

        X_dev = self.vectorize_tickets(dev_tickets, max_importe=max_importe)
        Y_dev = self.vectorize_categoria(dev_tickets)
        self.loss_fn = nn.CrossEntropyLoss()
        #self.loss_fn = nn.NLLLoss()
        for t in range(iterations):
            y_train = self._model(X_train)
            y_dev = self._model(X_dev)
            loss_train = self.loss_fn(y_train, torch.stack([i.argmax() for i in Y_train]))

            #loss_train = self.loss_fn(y_train, Y_train)
            #loss_dev = self.loss_fn(y_dev, torch.stack([i.argmax() for i in Y_dev]))
            #loss_dev = self.loss_fn(y_dev, Y_dev)
            if t % 100 == 99:
                print(t, loss_train.item())

            #loss_evo.loc[t] = [loss_train.item(), loss_dev.item()]
            optimizer.zero_grad()

            loss_train.backward()
            optimizer.step()

        self._model.training = False
        return loss_evo

    def predict(self, ticket, p):
        """
        Run the model over a contable.models.Ticket object
        :param ticket: contable.models.Ticket
        :param p: minimum probability to consider a category.
        :return: contable.models.Categoria object
        """
        x = self.vectorize_ticket(ticket)
        y = self._model(x)

        return self.devectorize_categoria(y, p)

    def save_nn(self, filename):
        conf = dict(
            in_dictionary=self.in_dictionary,
            out_dictionary=self.out_dictionary,
            model_in_keys=self.model_in_keys,
            model_out_key=self.model_out_key,
            dim_in=self.dim_in,
            dim_out=self.dim_out,
            dim_h=self.dim_h,
            layers=self.layers,
        )
        with open(filename + '.conf', 'wb') as f:
            pkl.dump(conf, f)

        torch.save(self._model, filename + '.pt')


    def load_conf(self, filename):
        with open(filename + '.conf', 'rb') as f:
            conf = pkl.load(f)
            print(conf)

        for key in ['in_dictionary',
                    'out_dictionary',
                    'model_in_keys',
                    'model_out_key',
                    'dim_in',
                    'dim_out',
                    'dim_h',
                    'layers']:
            try:
                self.__setattr__(key, conf[key])
            except KeyError:
                if key == 'model_out_key':
                    try:
                        self.__setattr__(key, conf['model_out_keys'])
                    except KeyError:
                        raise KeyError

    def load_nn(self, filename):
        self.load_conf(filename)

        self._init_nn(self.layers)
        self._model = torch.load(filename + '.pt')
        self._model.eval()

    def test_dataset(self, tickets, p):
        """
        Test a dataset and show how many predictions are correct
        :param tickets:
        :param p:
        :return: a thruple with the Hits, Miss and No Contest ratios over the dataset.
        """
        hits = 0
        miss = 0
        nc = 0
        for ticket in tickets:
            if self.predict(ticket, p).name == "No Categorizado":
                nc += 1
            elif self.predict(ticket, p).id == ticket.categoria.id:
                hits += 1
            else:
                miss += 1
        hits /= len(tickets)
        miss /= len(tickets)
        nc /= len(tickets)
        return hits, miss, nc

    def cost_dataset(self, tickets):
        """
        Return the loss function of a dataset.
        :param tickets: contable.models.Ticket objects
        :return:
        """
        Y = self.vectorize_categoria(tickets)
        X = self.vectorize_tickets(tickets)
        y_pred = self._model(X)
        loss = self.loss_fn(y_pred, Y)

        return loss


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
                con=[''.join(e for e in re.sub("\d+", "", con) if e.isalnum()) for con in ticket.concepto.split(' ')],
                importe_conv=ticket.importe * ticket.moneda.cambio,
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
        for i in range(layers - 1):
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
        # drop = nn.Dropout(p=0.3)
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
        torch.save(self._module.state_dict(), filename + '.torch')
        conf = dict(
            in_dictionary=self.in_dictionary,
            out_dictionary=self.out_dictionary,
            model_in_keys=self.model_in_keys,
            model_out_keys=self.model_out_key,
            dim_in=self.dim_in,
            dim_out=self.dim_out,
            dim_h=self.dim_h,
        )
        with open(filename + '.conf', 'wb') as f:
            pkl.dump(conf, f)

    def load_nn(self, filename):
        with open(filename + '.conf', 'rb') as f:
            conf = pkl.load(f)

        for key in ['in_dictionary', 'out_dictionary', 'model_in_keys', 'model_out_keys', 'dim_in', 'dim_out', 'dim_h']:
            self.__setattr__(key, conf[key])

        self._init_nn()
        self._module.load_state_dict(torch.load(filename + '.torch'))
        self._module.eval()

    def test_dataset(self, tickets, p):
        hits = 0
        miss = 0
        nc = 0
        for ticket in tickets:
            if self.predict(ticket, p).name == "No Categorizado":
                nc += 1
            elif self.predict(ticket, p).id == ticket.categoria.id:
                hits += 1
            else:
                miss += 1
        hits /= len(tickets)
        miss /= len(tickets)
        nc /= len(tickets)
        return hits, miss, nc


class Classifier(nn.Module):
    def __init__(self, dim_in, dim_out, dim_h, p=0.2, layers=5):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_h)
        self.fc2 = nn.ReLU()
        self.mid_funcs = []
        for layer in range(layers):
            self.mid_funcs.append((nn.Linear(dim_h, dim_h), nn.LeakyReLU()))
        self.fc5 = nn.Linear(dim_h, dim_out)
        self.fc6 = nn.LeakyReLU()
        self.fc7 = nn.LogSoftmax(dim=0)
        self.drop_layer = nn.Dropout(p=p)
        self.training = False

    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.fc2(out)
        for layer in self.mid_funcs:
            if self.training:
                out = self.drop_layer(out)
            out = layer[0](out)
            out = layer[1](out)
        if self.training:
            out = self.drop_layer(out)
        out = self.fc5(out)
        out = self.fc6(out)
        out = self.fc7(out)
        return out


def myCustomLoss(output, target):
    #specifying the batch size
    my_batch_size = output.size()[0]
    #calculating the log of softmax values
    my_outputs = F.log_softmax(output, dim=1)
    #selecting the values that correspond to labels
    my_outputs = my_outputs[range(my_batch_size), target]
    #returning the results
    return -torch.sum(my_outputs)/len(output)