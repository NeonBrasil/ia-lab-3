import numpy as np
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy.io import arff

data, meta = arff.loadarff('./bank.arff')

attributes = meta.names()
data_value = np.asarray(data)

age = np.asarray(data['age']).reshape(-1, 1)
job = np.asarray(data['job']).reshape(-1, 1)
marital = np.asarray(data['marital']).reshape(-1, 1)
education = np.asarray(data['education']).reshape(-1, 1)
default = np.asarray(data['default']).reshape(-1, 1)
average = np.asarray(data['average']).reshape(-1, 1)
housing = np.asarray(data['housing']).reshape(-1, 1)
loan = np.asarray(data['loan']).reshape(-1, 1)
contact = np.asarray(data['contact']).reshape(-1, 1)
day = np.asarray(data['day']).reshape(-1, 1)
month = np.asarray(data['month']).reshape(-1, 1)
duration = np.asarray(data['duration']).reshape(-1, 1)
campaign = np.asarray(data['campaign']).reshape(-1, 1)
pdays = np.asarray(data['pdays']).reshape(-1, 1)
previous = np.asarray(data['previous']).reshape(-1, 1)
poutcome = np.asarray(data['poutcome']).reshape(-1, 1)
subscribed = np.asarray(data['subscribed']).reshape(-1, 1)

le = LabelEncoder()
job = le.fit_transform(job.ravel()).reshape(-1, 1)
marital = le.fit_transform(marital.ravel()).reshape(-1, 1)
education = le.fit_transform(education.ravel()).reshape(-1, 1)
default = le.fit_transform(default.ravel()).reshape(-1, 1)
housing = le.fit_transform(housing.ravel()).reshape(-1, 1)
loan = le.fit_transform(loan.ravel()).reshape(-1, 1)
contact = le.fit_transform(contact.ravel()).reshape(-1, 1)
month = le.fit_transform(month.ravel()).reshape(-1, 1)
poutcome = le.fit_transform(poutcome.ravel()).reshape(-1, 1)
subscribed = le.fit_transform(subscribed.ravel())

features = np.concatenate((age, job, marital, education, default, average, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome), axis=1)
target = subscribed

Arvore = DecisionTreeClassifier(criterion='entropy').fit(features, target)

plt.figure(figsize=(10, 6.5))
tree.plot_tree(Arvore, feature_names=attributes[:-1], class_names=['no', 'yes'],
               filled=True, rounded=True)
plt.show()

fig, ax = plt.subplots(figsize=(25, 10))
metrics.ConfusionMatrixDisplay.from_estimator(Arvore, features, target, display_labels=['no', 'yes'], values_format='d', ax=ax)
plt.show()