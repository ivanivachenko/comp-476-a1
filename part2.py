import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Cython import inline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("drug200.csv")

'''Divide priors vs conditionals'''
prior = data.Drug
attr = data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]


'''#3 Plot distribution of instances for each class'''
'''data.groupby('Drug').size()'''

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
'''data['Sex'] = data['Sex'].apply(lambda x: 0 if x == "F" else 1)
data['BP'] = data['BP'].apply(lambda x: 1 if x == "LOW" else 2 if x == "NORMAL" else 3)
data['Cholesterol'] = data['Cholesterol'].apply(lambda x: 1 if x == "LOW" else 2 if x == "NORMAL" else 3)
'''

data.Sex = pd.get_dummies(data.Sex, dummy_na=False, columns=['Sex'], drop_first=False)
data.BP = pd.Categorical(data.BP, ['LOW', 'NORMAL', 'HIGH'], ordered=True)
data.BP = data.BP.cat.codes
data.Cholesterol = pd.Categorical(data.Cholesterol, ['NORMAL', 'HIGH'], ordered=True)
data.Cholesterol = data.Cholesterol.cat.codes

'''#5 Split Data set using default parameter values'''
x_train, x_test, y_train, y_test = train_test_split(prior, attr, test_size=0.2)


'''#6 Run 6 different classifiers'''

'''NB'''

model = GaussianNB()
model.fit([x_train, 5], y_train)


