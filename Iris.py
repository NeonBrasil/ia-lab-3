from sklearn.datasets import load_iris
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


data = load_iris()
features =data.data
target = data.target

Arvore = DecisionTreeClassifier(criterion = 'entropy').fit(features,target)

plt.figure(figsize=(10,6.5))
tree.plot_tree(Arvore,feature_names=data.feature_names,class_names=data.target_names,
                   filled=True, rounded=True)
plt.show()



Amostra = [[1.9 , 3 ,3.4, 0.2]]
class_Amostra = Arvore.predict(Amostra)

print(data.target_names[class_Amostra])


fig, ax = plt.subplots(figsize=(25, 10))
metrics.ConfusionMatrixDisplay.from_estimator(Arvore,features,target,display_labels=data.target_names, values_format='d', ax=ax)
plt.show()