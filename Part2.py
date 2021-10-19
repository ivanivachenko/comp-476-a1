import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz as eg
from sklearn.model_selection import GridSearchCV, ParameterGrid
import pydot
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support, plot_confusion_matrix, \
    accuracy_score

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
plt.title("Drugs Distribution")
plt.bar(plotCategories, plotValues)
plt.savefig("drug-distribution.pdf")

'''#4 Transform nominal FEATURES values to numerical values'''
data.Sex = pd.get_dummies(data.Sex, dummy_na=False, columns=['Sex'], drop_first=False)
data.BP = pd.Categorical(data.BP, ['LOW', 'NORMAL', 'HIGH'], ordered=True)
data.BP = data.BP.cat.codes
data.Cholesterol = pd.Categorical(data.Cholesterol, ['NORMAL', 'HIGH'], ordered=True)
data.Cholesterol = data.Cholesterol.cat.codes
data.Drug = pd.Categorical(data.Drug, ['drugA', 'drugB', 'drugC', 'drugX', 'drugY'], ordered=True)
data.Drug = data.Drug.cat.codes

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
x_train, x_test, y_train, y_test = train_test_split(attr, prior, test_size=0.2, random_state=42, shuffle=True)

with open("drugs-performance.txt", 'w') as f:
    f.write("DRUGS PERFORMANCE\n")
    f.close()

'''#6 Run 6 different classifiers'''

average_accuracy = [0] * 10
average_maf1 = [0] * 10
average_waf1 = [0] * 10

'''Function to write information for #7'''


def performance_results(ytest, pred, instance):
    results = precision_recall_fscore_support(y_test, pred, zero_division=1)
    accuracy = accuracy_score(y_test, pred)
    macrof1 = f1_score(ytest, pred, average='macro')
    weightedf1 = f1_score(ytest, pred, average='weighted')

    '''For #8'''
    average_accuracy[instance] = accuracy
    average_maf1[instance] = macrof1
    average_waf1[instance] = weightedf1

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


def calculate_and_write_averages(av_accuracy, av_micro, av_weighted):
    average_accuracy = np.mean(av_accuracy)
    average_microf1 = np.mean(av_micro)
    average_weightedf1 = np.mean(av_weighted)

    stddev_accuracy = np.std(av_accuracy)
    stddev_microf1 = np.std(av_micro)
    stddev_weightedf1 = np.std(av_weighted)

    with open("drugs-performance.txt", 'a') as f:
        f.write("\n\nAverage Accuracy of all 10: ")
        f.write(str(average_accuracy))
        f.write("\nAverage Micro-average F1 of all 10: ")
        f.write(str(average_microf1))
        f.write("\nAverage Weighted-average F1 of all 10: ")
        f.write(str(average_weightedf1))
        f.write("\n\nStandard Deviation of Accuracy of all 10: ")
        f.write(str(stddev_accuracy))
        f.write("\nStandard Deviation of Micro-average F1 of all 10: ")
        f.write(str(stddev_microf1))
        f.write("\nStandard Deviation of Weighted-average F1 of all 10: ")
        f.write(str(stddev_weightedf1))
        f.close()


'''NB'''
for x in range(10):
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    print(clf)
    predictions = clf.predict(x_test)

    '''#7 Analysis in text file'''

    with open("drugs-performance.txt", 'a') as f:
        f.write("\n\n*******************************************************\n")
        f.write("Description: GaussianNB model with default parameters")
        f.write("\nInstance #")
        f.write(str(x))
        f.write("\n\n Confusion Matrix \n")
        f.write(str(confusion_matrix(y_test, predictions)))
        f.close()

    performance_results(y_test, predictions, x)

calculate_and_write_averages(average_accuracy, average_maf1, average_waf1)

'''Base-DT'''

for x in range(10):
    clf = DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    print(clf)
    predictions = clf.predict(x_test)

    '''#7 Analysis in text file'''

    with open("drugs-performance.txt", 'a') as f:
        f.write("\n\n*******************************************************\n")
        f.write("Description: Base-DT model with default parameters")
        f.write("\nInstance #")
        f.write(str(x))
        f.write("\n\n Confusion Matrix \n")
        np.savetxt(f, confusion_matrix(y_test, predictions))
        f.close()

    performance_results(y_test, predictions, x)

calculate_and_write_averages(average_accuracy, average_maf1, average_waf1)

'''Top-DT'''

'''??Params??'''
for x in range(10):
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
        f.write("\nInstance #")
        f.write(str(x))
        f.write("\n\n Confusion Matrix \n")
        f.write(str(confusion_matrix(y_test, predictions)))
        f.close()

        performance_results(y_test, predictions, x)

calculate_and_write_averages(average_accuracy, average_maf1, average_waf1)

'''PER'''

for x in range(10):
    clf = Perceptron()
    clf.fit(x_train, y_train)
    print(clf)
    predictions = clf.predict(x_test)

    '''#7 Analysis in text file'''

    with open("drugs-performance.txt", 'a') as f:
        f.write("\n\n*******************************************************\n")
        f.write("Description: Perceptron model with default parameters")
        f.write("\nInstance #")
        f.write(str(x))
        f.write("\n\n Confusion Matrix \n")
        f.write(str(confusion_matrix(y_test, predictions)))
        f.close()

        performance_results(y_test, predictions, x)

calculate_and_write_averages(average_accuracy, average_maf1, average_waf1)

'''Base-MLP'''
for x in range(10):
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
        f.write("\nInstance #")
        f.write(str(x))
        f.write("\n\n Confusion Matrix \n")
        f.write(str(confusion_matrix(y_test, predictions)))
        f.close()

        performance_results(y_test, predictions, x)

calculate_and_write_averages(average_accuracy, average_maf1, average_waf1)

'''Top-MLP'''

for x in range(10):
    # mlpParams = {'activation': ('tanh', 'relu', 'identity'), 'hidden_layers': [10, 10, 10],
    #             'solver': ('adam', 'sgd')}

    # mlp = MLPClassifier()
    # mlpParams = {'solver': ('adam', 'sgd'), 'activation':  ('logistic', 'tanh', 'relu', 'identity'), 'hidden_layer_sizes': (10, 10, 10)}

    '''GRID = [
        {'estimator': [MLPClassifier()],
         'estimator__solver': ['adam', 'sgd'],
         'estimator__hidden_layer_sizes': [(10), (10), (10)],
         'estimator__activation': ['logistic', 'tanh', 'relu',
                                   'identity']
         }
    ]

    PIPELINE = Pipeline([('scaler', None), ('estimator', MLPClassifier())])'''

    # print(estimator.get_params().keys())
    # randomVar = MLPClassifier()
    # randomVar.fit(x_train, y_train)
    # grid_search_cv = GridSearchCV(estimator=PIPELINE, param_grid=GRID)
    # grid_search_cv.fit(clf, x_train, y_train)

    '''print(grid_search_cv.get_params().keys())
    print(grid_search_cv.best_estimator_)

   eg(
        grid_search_cv.best_estimator_,
        out_file="drug_topMLP.dot",
        feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'],
        class_names=['DrugA', 'DrugB', 'DrugC', 'DrugX', 'DrugY'],
        filled=True,
    )

    (graph,) = pydot.graph_from_dot_file('drug_topMLP.dot')
    graph.write_png('drug_topMLP.png')

    predictions = grid_search_cv.predict(x_test)

    with open("drugs-performance.txt", 'a') as f:
        f.write("\n\n*******************************************************\n")
        f.write("Description: Top-MLP model with parameters"
                " \nactivation = [sigmoid, tanh, relu, identity]"
                "\nhidden_layers = a range between 3 and 10"
                "\nsolver = [adam, sgd]")
        f.write("\nInstance #")
        f.write(str(x))
        f.write("\n\n Confusion Matrix \n")
        np.savetxt(f, confusion_matrix(y_test, predictions))
        f.close()

        # plot_confusion_matrix(grid_search_cv, x_test, y_test)
        # plt.title("Top MLP Confusion Matrix")
        # plt.savefig("Top-MLPConfusionMatrix.png")

        performance_results(y_test, predictions, x)

calculate_and_write_averages(average_accuracy, average_maf1, average_waf1)'''
