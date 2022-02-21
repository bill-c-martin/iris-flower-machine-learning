# Note: Run this through index.ipynb Jupyter Notebook to see generated graphs

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load iris flower dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
print('Iris flower dataset loaded.', end='\n\n')

# Review dataset
print('Dataset has rows and cols:', dataset.shape, end='\n\n')

print('Dataset 5-row preview:')
print(dataset.head(5), end='\n\n')

print('Dataset stat summary:')
print(dataset.describe(), end='\n\n')

print('Dataset class distribution:')
print(dataset.groupby('class').size(), end='\n\n')

# Box and whisker plots
print('Box and whisker plots for the dataset:')
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

print("Histograms of dataset, with sepal length and width showing Gaussian distributions:")
dataset.hist()
pyplot.show()

print("Scatterplots of attrbutes, with diagonal groupings indicating high correlations and predictable relationships:")
scatter_matrix(dataset)
pyplot.show()