import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

# Generating 3D make-moons data
def make_moons_3d(n_samples=500, noise=0.1):
    # Generate the original 2D make_moons data
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  # Adding a sinusoidal variation in the third dimension

    # Concatenating the positive and negative moons with an offset and noise
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # Adding Gaussian noise
    X += np.random.normal(scale=noise, size=X.shape)

    return X, y

def Plotting(x_tr,x_te,y_tr,y_te,title1,title2):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121, projection='3d')
    scatter = ax.scatter(x_tr[:, 0], x_tr[:, 1], x_tr[:, 2], c=y_tr, cmap='viridis', marker='o')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title1)

    ax = fig.add_subplot(122, projection='3d')
    scatter = ax.scatter(x_te[:, 0], x_te[:, 1], x_te[:, 2], c=y_te, cmap='viridis', marker='o')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title2)
    plt.show()

# Generate the training & test data (1000 datapoints)
X_tr, labels_tr = make_moons_3d(n_samples=500, noise=0.2)
X_te, labels_te = make_moons_3d(n_samples=250, noise=0.2)

Plotting(X_tr,X_te,labels_tr,labels_te,'Training 3D Make Moons','Test 3D Make Moons')

########decision tree#########
depth = 5
clf = DecisionTreeClassifier(max_depth=depth,random_state=50)
clf.fit(X_tr, labels_tr)

DTy_train_pred = clf.predict(X_tr)
DTy_test_pred = clf.predict(X_te)

Plotting(X_tr,X_te,DTy_train_pred,DTy_test_pred,'Training result of DT','Test result of DT')

training_errors = 1 - clf.score(X_tr, labels_tr)
test_errors = 1 - clf.score(X_te, labels_te)
print(f"DT训练误差: {training_errors:.8f}")
print(f"DT测试误差: {test_errors:.8f}")

########Adaboost + decision tree#########
base_clf = DecisionTreeClassifier(max_depth=depth,random_state=50)
adaboost_clf = AdaBoostClassifier(
    estimator=base_clf,
    n_estimators=15,
    random_state=50)
adaboost_clf.fit(X_tr, labels_tr)

ADTy_train_pred = adaboost_clf.predict(X_tr)
ADTy_test_pred = adaboost_clf.predict(X_te)

Plotting(X_tr,X_te,ADTy_train_pred,ADTy_test_pred,'Training result of ADT','Test result of ADT')

training_errors = 1 - adaboost_clf.score(X_tr, labels_tr)
test_errors = 1 - adaboost_clf.score(X_te, labels_te)
print(f"adaboost+DT训练误差: {training_errors:.8f}")
print(f"adaboost+DT测试误差: {test_errors:.8f}")

##########SVM##########
#RBF
svm_rbf_clf = SVC(kernel='rbf',gamma='scale',random_state=50)
svm_rbf_clf.fit(X_tr, labels_tr)
SVMRBFy_train_pred = svm_rbf_clf.predict(X_tr)
SVMRBFy_test_pred = svm_rbf_clf.predict(X_te)

Plotting(X_tr,X_te,SVMRBFy_train_pred,SVMRBFy_test_pred,'Training result of SVM+RBF','Test result of SVM+RBF')

training_errors = 1 - svm_rbf_clf.score(X_tr, labels_tr)
test_errors = 1 - svm_rbf_clf.score(X_te, labels_te)
print(f"SVM+RBF训练误差: {training_errors:.8f}")
print(f"SVM+RBF测试误差: {test_errors:.8f}")

#linear kernel
svm_lk_clf = SVC(kernel='linear',random_state=50)
svm_lk_clf.fit(X_tr, labels_tr)
SVMlky_train_pred = svm_lk_clf.predict(X_tr)
SVMlky_test_pred = svm_lk_clf.predict(X_te)

Plotting(X_tr,X_te,SVMlky_train_pred,SVMlky_test_pred,'Training result of SVM+linear kernel','Test result of SVM+linear kernel')

training_errors = 1 - svm_lk_clf.score(X_tr, labels_tr)
test_errors = 1 - svm_lk_clf.score(X_te, labels_te)
print(f"SVM+lk训练误差: {training_errors:.8f}")
print(f"SVM+lk测试误差: {test_errors:.8f}")

#polynomial kernel
svm_poly_clf = SVC(kernel='poly',degree=3,random_state=50)
svm_poly_clf.fit(X_tr, labels_tr)
SVMpoy_train_pred = svm_poly_clf.predict(X_tr)
SVMpoy_test_pred = svm_poly_clf.predict(X_te)

Plotting(X_tr,X_te,SVMpoy_train_pred,SVMpoy_test_pred,'Training result of SVM+polynomial kernel','Test result of SVM+polynomial kernel')

training_errors = 1 - svm_poly_clf.score(X_tr, labels_tr)
test_errors = 1 - svm_poly_clf.score(X_te, labels_te)
print(f"SVM+poly训练误差: {training_errors:.8f}")
print(f"SVM+poly测试误差: {test_errors:.8f}")