import numpy as np
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from scipy.io import arff

data, meta = arff.loadarff('./TennisMatch.arff')

attributes = meta.names()
data_value = np.asarray(data)

outlook = np.asarray(data['Outlook']).reshape(-1, 1)
temperature = np.asarray(data['Temperature']).reshape(-1, 1)
humidity = np.asarray(data['Humidity']).reshape(-1, 1)
wind = np.asarray(data['Wind']).reshape(-1, 1)
play_tennis = np.asarray(data['PlayTennis']).reshape(-1, 1)

le = LabelEncoder()
outlook = le.fit_transform(outlook.ravel()).reshape(-1, 1)
temperature = le.fit_transform(temperature.ravel()).reshape(-1, 1)
humidity = le.fit_transform(humidity.ravel()).reshape(-1, 1)
wind = le.fit_transform(wind.ravel()).reshape(-1, 1)
play_tennis = le.fit_transform(play_tennis.ravel())

features = np.concatenate((outlook, temperature, humidity, wind), axis=1)
target = play_tennis

Arvore = DecisionTreeClassifier(criterion='entropy').fit(features, target)

plt.figure(figsize=(10, 6.5))
tree.plot_tree(Arvore, feature_names=['Outlook', 'Temperature', 'Humidity', 'Wind'], class_names=['No', 'Yes'],
               filled=True, rounded=True)
plt.show()

fig, ax = plt.subplots(figsize=(25, 10))
metrics.ConfusionMatrixDisplay.from_estimator(Arvore, features, target, display_labels=['No', 'Yes'], values_format='d', ax=ax)
plt.show()