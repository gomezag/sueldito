from django.apps import AppConfig
from django.db.models import Q
from contable.models import *
from .nn import AutoClassifier, CategoriaClassifier

class NnClasifyConfig(AppConfig):
    name = 'nn_clasify'

def train_classifier(categoria,train_n=100, neurons=30, iterations=1000, lr=1e-3, layers=1):
    classifier = CategoriaClassifier()
    train_tickets, test_tickets = build_train_data_single_cat_normalized(categoria, train_n=train_n)
    classifier.setup_train_data(categoria)
    classifier.train_nn(train_tickets, neurons=neurons, iterations=iterations, lr=lr, layers=layers)
    print('Train: ', classifier.test_dataset(train_tickets, 0.6))
    print('Test: ',classifier.test_dataset(test_tickets, 0.6))

    return classifier

def build_train_data_normalize_categories(train_n=1000, test_n=300):
    tickets = Ticket.objects.filter(cuenta=Cuenta.objects.get(name="BBVA Es - C.C."))
    # Normalize distribution of categories among the samples.
    # Take train_n as maximum number of train set
    cat_tickets = []
    for categoria in Categoria.objects.all():
        cat_ticket = tickets.filter(categoria=categoria)
        if len(cat_ticket)>10:
            cat_tickets.append(cat_ticket)
    max_cat = min([len(i) for i in cat_tickets]+[train_n])
    train_tickets = []
    # Select max_cat items of each category
    for cat in cat_tickets:
        train_tickets += cat.order_by('?')[:max_cat]
    # Don't try to select more tickets than available
    test_n = max([test_n, len(tickets)])
    test_tickets = tickets.order_by('?')[:test_n-1]

    return train_tickets, test_tickets

def build_train_data_single_cat_normalized(categoria, train_n=100, test_n=300):
    tickets = Ticket.objects.filter(cuenta=Cuenta.objects.get(name="BBVA Es - C.C."))
    # Normalize distribution of categories among the samples.
    # Take train_n as maximum number of train set
    cat_tickets = tickets.filter(categoria=categoria).order_by('?')
    cat_tickets = cat_tickets[:train_n]
    other_tickets = tickets.filter(~Q(categoria=categoria)).order_by('?')
    other_tickets = other_tickets[:min([len(other_tickets), len(cat_tickets)])]
    train_tickets = [ticket for ticket in cat_tickets]+other_tickets

    # Don't try to select more tickets than available
    test_n = max([test_n, len(tickets)])
    test_tickets = tickets.order_by('?')[:test_n-1]
    print(len(train_tickets), len(test_tickets))
    return train_tickets, test_tickets

def build_train_data(train_n=0.8, test_n=0.2, dev_n=0):
    tickets = Ticket.objects.filter(cuenta=Cuenta.objects.get(name="BBVA Es - C.C."))
    tickets = tickets.order_by('?')
    train_n = int(len(tickets)*train_n)
    train_tickets = tickets[:train_n]
    test_tickets = tickets[train_n:]
    return train_tickets, test_tickets


def retrain_classifier(classifier, train_n=1000, iterations=100, mini_iterations=10, lr=1e-4):
    for i in range(int(iterations)):
        train_tickets, test_tickets = build_train_data_single_cat_normalized(classifier.categoria, train_n=train_n)
        classifier.train_nn(train_tickets, iterations=mini_iterations, lr=lr)
        print(classifier.test_dataset(train_tickets, 0.5))
        if i%10 == 9:
            print('Test #'+str(i)+': ',classifier.test_dataset(test_tickets, 0.5))