from nn_clasify.apps import train_classifier, build_train_data, retrain_classifier
#classifier = train_classifier(train_n=0.8, neurons=40, iterations=10, lr=1e-1, layers=5)
from contable.models import Categoria, Ticket
from nn_clasify.apps import train_classifier, build_train_data, retrain_it, save_train_sets, load_train_sets
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1)
train, dev, test = build_train_data()
loss_evo, classifier = train_classifier(train, dev, test, neurons=80, iterations=500, lr=1e-1, layers=40, dropout=False)
classifier.save_nn('./torch_models/testmodel')
from nn_clasify.nn import AutoClassifier
classifier = AutoClassifier()
classifier.load_nn('./torch_models/nllloss')
ax.plot(loss_evo["train"], label="train")
ax.plot(loss_evo["dev"], label="dev")
ax.legend(loc="upper right")
fig.show()
loss_evo = retrain_it(classifier, train, dev, test, iterations=3000, lr=1e-1, dropout=False)
ax.plot(loss_evo["train"], label="train")
ax.plot(loss_evo["dev"], label="dev")
fig.show()
print('Train: ', classifier.test_dataset(train, -3))
print('Dev: ', classifier.test_dataset(dev, -3))
print('Test: ', classifier.test_dataset(test, -3))
classifier = train_classifier(Categoria.objects.get(name="Comida&Super"),train_n=100, neurons=40, iterations=10, lr=1e-1, layers=5)
classifier = train_classifier('all',train_n=100, neurons=40, iterations=10, lr=1e-1, layers=5)
retrain_classifier(classifier, train_n=100, iterations=1000, mini_iterations=1, lr=1e-1)

from nn_clasify.apps import retrain_classifier
from nn_clasify.nn import AutoClassifier
classifier = AutoClassifier()
classifier.load_nn('torch_models/minibatch')

retrain_classifier(classifier, train_n=10, iterations=100, mini_iterations=1000, lr=1e-2)

classifier.save_nn('torch_models/minibatch')
from contable.models import *
ticket = Ticket.objects.filter(cuenta=Cuenta.objects.get(name="BBVA Es - C.C."))[120]
