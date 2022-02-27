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
print('Box and whisker plots for the dataset, showing the distribution for each measurement:')
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# Histograms
print("Histograms of dataset, with sepal length and width showing Gaussian distributions:")
dataset.hist()
pyplot.show()

# Scatterplots
print("Scatterplots of attrbutes, with diagonal groupings indicating high correlations and predictable relationships:")
scatter_matrix(dataset)
pyplot.show()

# Split out validation dataset into:
#   x: 2d array of flower measurements
#   y: 1d array of flower names
array = dataset.values
x = array[:,0:4]
y = array[:,4]

# 80% of data is for training, remaining 20% for validation
X_train, X_validation, Y_train, Y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

# Build up some spot-checking models
models = []
# Linear
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
# Nonlinear
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Evaluate each model using stratified 10-fold cross validation, to select the best one
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    # print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare algorithms on box and whisker plots
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm % Accuracy Comparison')
print('Linear Discriminant Analysis (LDA) and Support Vector Machines (SVM) achieved 100% accuracy on the Iris Flowers dataset. Whereas, Logistic Regression (LR), Decision Tree Classifier(CART), and Gaussian (NB) push down as low as the 80% range:')
pyplot.show()

# Make predictions on validation dataset, using SVM
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
print('Model selected: Support Vector Machines(SVM)')
predictions = model.predict(X_validation)

print('\nAccuracy Score:', end='\n')
print(accuracy_score(Y_validation, predictions))

print('\nConfusion Matrix, showing indications of any errors made:', end='\n')
print(confusion_matrix(Y_validation, predictions))

print('\nClassification Report, showing excellent results for each class:', end='\n')
print(classification_report(Y_validation, predictions))