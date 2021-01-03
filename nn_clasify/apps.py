from django.apps import AppConfig
from django.db.models import Q
from contable.models import *
import collections as col
import numpy as np
import pandas as pd
import pickle as pkl
import nn_clasify.nn as nn
import os

class NnClasifyConfig(AppConfig):
    name = 'nn_clasify'


def train_classifier(classifier, train_tickets, dev_tickets, test_tickets, categoria='all', train_n=100,
                     neurons=30, iterations=1000, lr=1e-3, layers=4, dropout=True):
    if categoria == 'all':
        classifier.setup_train_data(train_tickets)
        loss_evo = classifier.train_nn(train_tickets, dev_tickets, neurons=neurons, iterations=iterations, lr=lr
                                       , dropout=dropout)
        print('Train: ', classifier.test_dataset(train_tickets, -0.6))
        print('Dev: ', classifier.test_dataset(dev_tickets, -0.6))
        print('Test: ', classifier.test_dataset(test_tickets, -0.6))

        return loss_evo, classifier
    return 'Select all for categoria'


def build_train_data_normalize_categories(train_n=1000, test_n=300):
    tickets = Ticket.objects.filter(cuenta=Cuenta.objects.get(name="BBVA Es - C.C."))
    # Normalize distribution of categories among the samples.
    # Take train_n as maximum number of train set
    cat_tickets = []
    for categoria in Categoria.objects.all():
        cat_ticket = tickets.filter(categoria=categoria)
        if len(cat_ticket) >= train_n * 0.15:
            cat_tickets.append(cat_ticket)
    max_cat = min([len(i) for i in cat_tickets] + [train_n])
    train_tickets = []
    # Select max_cat items of each category
    for cat in cat_tickets:
        train_tickets += cat.order_by('?')[:max_cat]
    # Don't try to select more tickets than available
    test_n = max([test_n, len(tickets)])
    test_tickets = tickets.order_by('?')[:test_n - 1]

    return train_tickets, test_tickets


def build_train_data_single_cat_normalized(categoria, train_n=100, test_n=300):
    tickets = Ticket.objects.filter(cuenta=Cuenta.objects.get(name="BBVA Es - C.C."))
    # Normalize distribution of categories among the samples.
    # Take train_n as maximum number of train set
    cat_tickets = tickets.filter(categoria=categoria).order_by('?')
    cat_tickets = cat_tickets[:train_n]
    other_tickets = tickets.filter(~Q(categoria=categoria)).order_by('?')
    other_tickets = other_tickets[:min([len(other_tickets), len(cat_tickets)])]
    train_tickets = [ticket for ticket in cat_tickets] + other_tickets

    # Don't try to select more tickets than available
    test_n = max([test_n, len(tickets)])
    test_tickets = tickets.order_by('?')[:test_n - 1]
    print(len(train_tickets), len(test_tickets))
    return train_tickets, test_tickets


def build_train_data(train_n=0.8, test_n=0.1, dev_n=0.1):
    """
    Split the data in the database
    :param train_n:
    :param test_n:
    :param dev_n:
    :return:
    """
    tickets = Ticket.objects.filter(cuenta=Cuenta.objects.get(name="BBVA Es - C.C."))
    tickets = tickets.order_by('?')
    train_tickets = []
    dev_tickets = []
    test_tickets = []

    #Counter = col.namedtuple('Category', ['id', 'count'])
    train_categories_n = np.array([[cat.id, 0] for cat in Categoria.objects.all().order_by('id')])
    dev_categories_n = np.array([[cat.id, 0] for cat in Categoria.objects.all().order_by('id')])
    test_categories_n = np.array([[cat.id, 0] for cat in Categoria.objects.all().order_by('id')])
    for ticket in tickets:
        k = [cat[0] for cat in train_categories_n].index(ticket.categoria.id)
        a = np.array([train_categories_n[k][1], dev_categories_n[k][1], test_categories_n[k][1]])
        if any(a == 0):
            # If any set still doesn't have a ticket with this category, throw it there
            i = a.tolist().index(0)
        else:
            # If not, we want to keep the percentages that we want for the total number of tickets in each set,
            # and maintain a similarity between the category percentages within them.
            # We start by setting the % of a category within each dataset as a satisficing value of 5% difference,
            # and then we optimize for the datasets.pkl split percentage.
            a = [
                train_categories_n[k][1]/len(train_tickets),
                dev_categories_n[k][1]/len(dev_tickets),
                test_categories_n[k][1]/len(test_tickets)
            ]
            # Check the difference with the maximum category percentage achieved in each set,
            a = np.array([i/max(a) for i in a])
            if any(a < 0.90):
                # If some percentage has a 5% (or more) difference with the maximum one in all sets, then we assign
                # it to that set
                i = a.tolist().index(min(a))
            else:
                # Else, we want to optimize the percentages asked.
                total_tickets = len(train_tickets)+len(test_tickets)+len(dev_tickets)
                a = np.array([
                    len(train_tickets)/(total_tickets*train_n),
                    len(dev_tickets)/(total_tickets*dev_n),
                    len(test_tickets)/(total_tickets*test_n),
                ])
                i = a.tolist().index(min(a))

        if i == 0:
            train_categories_n[k][1] += 1
            train_tickets.append(ticket)
        elif i == 1:
            dev_categories_n[k][1] += 1
            dev_tickets.append(ticket)
        elif i == 2:
            test_categories_n[k][1] += 1
            test_tickets.append(ticket)

    print("{:.2f}%".format(len(train_tickets)/len(tickets)), "{:.2f}%".format(len(dev_tickets)/len(tickets)), "{:.2f}%".format(100*len(test_tickets)/len(tickets)))
    results = pd.DataFrame(columns=[cat.name for cat in Categoria.objects.all()], index=["train", "test", "dev"])
    results.loc["train"]=[cat[1] for cat in train_categories_n]
    results.loc["test"] = [cat[1] for cat in test_categories_n]
    results.loc["dev"] = [cat[1] for cat in dev_categories_n]
    print(results)
    return train_tickets, dev_tickets, test_tickets

def retrain_it(classifier, train_tickets, dev_tickets, test_tickets, iterations=1000, lr=1e-4, dropout=True):
    loss_evo = classifier.train_nn(train_tickets, dev_tickets, iterations=iterations, lr=lr, dropout=dropout)
    print('Train: ', classifier.test_dataset(train_tickets, -0.6))
    print('Dev: ', classifier.test_dataset(dev_tickets, -0.6))
    print('Test: ', classifier.test_dataset(test_tickets, -0.6))
    return loss_evo

def retrain_classifier(classifier, train, dev, test, iterations=100, lr=1e-4):
    # if isinstance(classifier, CategoriaClassifier):
    #     train_tickets, test_tickets = build_train_data_single_cat_normalized(classifier.categoria, train_n=train_n)
    #     for i in range(int(iterations)):
    #         classifier.train_nn(train_tickets, iterations=iterations, lr=lr)
    #          print(classifier.test_dataset(train_tickets, 0.5))
    #         if i % 100 == 9:
    #             print('Test #' + str(i) + ': ', classifier.cost_dataset(test_tickets))
    if isinstance(classifier, nn.AutoClassifier):
        #train_tickets, test_tickets = build_train_data_normalize_categories(train_n=train_n)

        loss_evo = classifier.train_nn(train, dev, iterations=iterations, lr=lr)
        # print(classifier.test_dataset(train_tickets, 0.5))
        print('Train: ', classifier.test_dataset(train, 0.6))
        print('Dev: ', classifier.test_dataset(dev, 0.6))
        print('Test: ', classifier.test_dataset(test, 0.6))

        return loss_evo


def save_train_sets(train, dev, test):
    ids = dict(train=[tic.id for tic in train], dev=[tic.id for tic in dev], test=[tic.id for tic in test])
    with open('./torch_models/datasets.pkl', 'wb') as file:
        pkl.dump(ids, file)


def load_train_sets():
    fname = './torch_models/datasets.pkl'
    with open(fname, 'rb') as file:
        ids = pkl.load(file)
    train = [Ticket.objects.get(id=i) for i in ids['train']]
    dev = [Ticket.objects.get(id=i) for i in ids['dev']]
    test = [Ticket.objects.get(id=i) for i in ids['test']]

    return train, dev, test

def load_nn(filename, classifier):
    classifier.load_nn(filename)

    return classifier