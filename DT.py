import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# Load features and labels
features = np.load('features.npy')
labels = np.load('labels.npy')
X = np.array(features)
y = np.array(labels)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize classifiers
decision_tree_model = DecisionTreeClassifier()
random_forest_model = RandomForestClassifier(random_state=42,max_depth=10)
svm_model = SVC(gamma=2, C=1, random_state=42)
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=42)
knn_model = KNeighborsClassifier()
ada_boost_model = AdaBoostClassifier()
from sklearn.gaussian_process.kernels import RBF
G_model =GaussianProcessClassifier(kernel = 1.0 * RBF(1.0), random_state=42),
logistic_regression_model = LogisticRegression()
lda_model = LinearDiscriminantAnalysis()

# Train the models
# decision_tree_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)
# svm_model.fit(X_train, y_train)
# mlp_model.fit(X_train, y_train)
# knn_model.fit(X_train, y_train)
# ada_boost_model.fit(X_train, y_train)
# G_model.fit(X_train, y_train)
# logistic_regression_model.fit(X_train, y_train)
# lda_model.fit(X_train, y_train)



# Predict and evaluate the models
# dt_accuracy = accuracy_score(y_test, decision_tree_model.predict(X_test))
rf_accuracy = accuracy_score(y_test, random_forest_model.predict(X_test))
# svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test))
# mlp_accuracy = accuracy_score(y_test, mlp_model.predict(X_test))
# knn_accuracy = accuracy_score(y_test, knn_model.predict(X_test))
# ada_boost_accuracy = accuracy_score(y_test, ada_boost_model.predict(X_test))
# G_accuracy = accuracy_score(y_test, G_model.predict(X_test))
# logistic_regression_accuracy = accuracy_score(y_test, logistic_regression_model.predict(X_test))
# lda_accuracy = accuracy_score(y_test, lda_model.predict(X_test))

# print(f"Decision Tree Accuracy: {dt_accuracy}")
print(f"Random Forest Accuracy: {rf_accuracy}")
# print(f"SVM Accuracy: {svm_accuracy}")
# print(f"MLP Neural Network Accuracy: {mlp_accuracy}")
# print(f"KNN Accuracy: {knn_accuracy}")
# print(f"AdaBoost Accuracy: {ada_boost_accuracy}")
# print(f"Gaussian Process Accuracy: {G_accuracy}")
# print(f"Logistic Regression Accuracy: {logistic_regression_accuracy}")
# print(f"LDA Accuracy: {lda_accuracy}")
