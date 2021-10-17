import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz as eg
from sklearn.model_selection import GridSearchCV
import pydot
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support, plot_confusion_matrix, accuracy_score
import string


data = pd.read_csv("drug200.csv")

'''#3 Plot distribution of instances for each class'''

'''nb of instances for each class'''
drugACount = data.Drug.value_counts().drugA
drugBCount = data.Drug.value_counts().drugB
drugCCount = data.Drug.value_counts().drugC
drugXCount = data.Drug.value_counts().drugX
drugYCount = data.Drug.value_counts().drugY


'''Put it in a chart'''
plotValues = [drugACount, drugBCount, drugCCount, drugXCount, drugYCount]
plotCategories = ['Drug A', 'Drug B', 'Drug C', 'Drug X', 'Drug Y']
plt.bar(plotCategories, plotValues)
plt.savefig("drug-distribution.pdf")


'''#4 Transform nominal FEATURES values to numerical values'''
data.Sex = pd.get_dummies(data.Sex, dummy_na=False, columns=['Sex'], drop_first=False)
data.BP = pd.Categorical(data.BP, ['LOW', 'NORMAL', 'HIGH'], ordered=True)
data.BP = data.BP.cat.codes
data.Cholesterol = pd.Categorical(data.Cholesterol, ['NORMAL', 'HIGH'], ordered=True)
data.Cholesterol = data.Cholesterol.cat.codes


'''Divide priors vs conditionals'''
prior = data.Drug
attr = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
'''x should be features
y should be f(x)'''
'''attr'''
X = np.array(attr)
'''prior'''
Y = np.array(prior)
'''#5 Split Data set using default parameter values'''
x_train, x_test, y_train, y_test = train_test_split(attr, prior, test_size=0.2)

'''#6 Run 6 different classifiers'''


def performance_results(ytest, pred):
    results = precision_recall_fscore_support(y_test, pred)
    print(results)

    accuracy = accuracy_score(y_test, pred)
    print(accuracy)

    macrof1 = f1_score(ytest, pred, average='macro')
    weightedf1 = f1_score(ytest, pred, average='weighted')

    with open("drugs-performance.txt", 'a') as f:
        f.write('\n\nPrecision: \n')
        f.write('\nDrug A: ')
        f.write(str(results[0][0]))
        f.write('\nDrug B: ')
        f.write(str(results[0][1]))
        f.write('\nDrug C: ')
        f.write(str(results[0][2]))
        f.write('\nDrug X: ')
        f.write(str(results[0][3]))
        f.write('\nDrug Y: ')
        f.write(str(results[0][4]))
        f.write('\n\nRecall: \n')
        f.write('\nDrug A: ')
        f.write(str(results[1][0]))
        f.write('\nDrug B: ')
        f.write(str(results[1][1]))
        f.write('\nDrug C: ')
        f.write(str(results[1][2]))
        f.write('\nDrug X: ')
        f.write(str(results[1][3]))
        f.write('\nDrug Y: ')
        f.write(str(results[1][4]))
        f.write('\n\nF1-score: \n')
        f.write('\nDrug A: ')
        f.write(str(results[2][0]))
        f.write('\nDrug B: ')
        f.write(str(results[2][1]))
        f.write('\nDrug C: ')
        f.write(str(results[2][2]))
        f.write('\nDrug X: ')
        f.write(str(results[2][3]))
        f.write('\nDrug Y: ')
        f.write(str(results[2][4]))
        f.write('\n\nAccuracy: ')
        f.write(str(accuracy))
        f.write('\n\nMacro-Average F1: ')
        f.write(str(macrof1))
        f.write('\n\nWeigted-Average F1: ')
        f.write(str(weightedf1))
        f.close()


'''NB'''

clf = GaussianNB()
clf.fit(x_train, y_train)
print(clf)
predictions = clf.predict(x_test)


'''#7 Analysis in text file'''

with open("drugs-performance.txt", 'w') as f:
    f.write("*******************************************************\n")
    f.write("Description: GaussianNB model with default parameters")
    f.close()

    print(confusion_matrix(y_test, predictions))
    '''Save in txt????'''
    plot_confusion_matrix(clf, x_test, y_test)
    plt.savefig("GaussianConfusionMatrix.png")
    plt.show()

    performance_results(y_test, predictions)



'''Bse-DT'''

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
print(clf)
predictions = clf.predict(x_test)

'''#7 Analysis in text file'''

with open("drugs-performance.txt", 'a') as f:
    f.write("\n\n*******************************************************\n")
    f.write("Description: Base-DT model with default parameters")
    f.close()

    print(confusion_matrix(y_test, predictions))
    '''Save in txt????'''
    plot_confusion_matrix(clf, x_test, y_test)
    plt.savefig("BaseDTConfusionMatrix.png")
    plt.show()

    performance_results(y_test, predictions)

'''Top-DT'''

'''??Params??'''
params = {'criterion': ['gini', 'entropy'], 'max_depth': list(range(2, 10)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(), params)
grid_search_cv.fit(x_train, y_train)

print(grid_search_cv.best_estimator_)

eg(
    grid_search_cv.best_estimator_,
    out_file=("drug_topDT.dot"),
    feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'],
    class_names=['DrugA', 'DrugB', 'DrugC', 'DrugX', 'DrugY'],
    filled=True,
)

(graph,) = pydot.graph_from_dot_file('drug_topDT.dot')
graph.write_png('drug_topDT.png')

predictions = grid_search_cv.predict(x_test)

'''#7 Analysis in text file'''

with open("drugs-performance.txt", 'a') as f:
    f.write("\n\n*******************************************************\n")
    f.write("Description: Top-DT model with parameters"
            " \ncriterion = [gini, entropy]"
            "\nmax-depth = a range between 2 and 10"
            "\nmin_samples_split = [2, 3, 4]")
    f.close()

    print(confusion_matrix(y_test, predictions))
    '''Save in txt????'''
    plot_confusion_matrix(grid_search_cv, x_test, y_test)
    plt.savefig("Top-DTConfusionMatrix.png")
    plt.show()

    performance_results(y_test, predictions)

'''PER'''

clf = Perceptron()
clf.fit(x_train, y_train)
print(clf)
predictions = clf.predict(x_test)

'''#7 Analysis in text file'''

with open("drugs-performance.txt", 'a') as f:
    f.write("\n\n*******************************************************\n")
    f.write("Description: Perceptron model with default parameters")
    f.close()

    print(confusion_matrix(y_test, predictions))
    '''Save in txt????'''
    plot_confusion_matrix(clf, x_test, y_test)
    plt.savefig("PERConfusionMatrix.png")
    plt.show()

    performance_results(y_test, predictions)

'''Base-MLP'''
mlpcParams = {'hidden_layer_sizes': [1, 100], 'activation': ['sigmoid', 'logistic'], 'solver': 'sgd'}
clf = MLPClassifier(hidden_layer_sizes=[1, 100], activation='logistic', solver='sgd')
clf.fit(x_train, y_train)
print(clf)
predictions = clf.predict(x_test)


'''#7 Analysis in text file'''

with open("drugs-performance.txt", 'a') as f:
    f.write("\n\n*******************************************************\n")
    f.write("Description: Base-MLP model with parameters\n")
    f.write("hidden_layer_sizes = [1, 100]\n")
    f.write("activation = [sigmoid, logistic]\n")
    f.write("solver = sdg")
    f.close()

    print(confusion_matrix(y_test, predictions))
    '''Save in txt????'''
    plot_confusion_matrix(clf, x_test, y_test)
    plt.savefig("BaseMLPConfusionMatrix.png")
    plt.show()

    performance_results(y_test, predictions)



'''Top-MLP'''


'''??Params??
mlpParams = {'activation': ['sigmoid', 'tanh', 'relu', 'identity'], 'hidden_layers': [3, 10], 'solver': ['adam', 'sgd']}
grid_search_cv = GridSearchCV(MLPClassifier())
grid_search_cv.fit(x_train, y_train)

print(grid_search_cv.get_params().keys())'''
'''print(grid_search_cv.best_estimator_)

eg(
    grid_search_cv.best_estimator_,
    out_file="drug_topMLP.dot",
    feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'],
    class_names=['DrugA', 'DrugB', 'DrugC', 'DrugX', 'DrugY'],
    filled=True,
)

(graph,) = pydot.graph_from_dot_file('drug_topMLP.dot')
graph.write_png('drug_topMLP.png')'''




