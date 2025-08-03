import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import GUI

IRIS_POISONED_POINTS = np.array([[5.0, 3.2], [5.1, 3.3], [5.2, 3.4]])
WINE_POISONED_POINTS = np.array([[13, 2], [13.5, 1.6], [14.2, 1.7]])


def create_model(is_iris):
    """
    this function creates the DBP model with SVC using sklearn package
    :param is_iris: if the data set is iris or wine
    :return:
    """
    if is_iris:
        dataset = datasets.load_iris()
    else:
        dataset = datasets.load_wine()
    x = dataset.data[:, :2]  # takes only 2 first features in order to be in 2D
    y = dataset.target
    x = x[y != 2]  # remains only 2 classes
    y = y[y != 2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, shuffle=True)
    model = SVC(kernel='linear')
    model.fit(x_train, y_train)
    if is_iris:
        poisoned_point, poisoned_label = IRIS_POISONED_POINTS, np.array([1, 1, 1])
    else:
        poisoned_point, poisoned_label = WINE_POISONED_POINTS, np.array([1, 1, 1])
    x_poisoned = np.vstack([x_train, poisoned_point])
    y_poisoned = np.hstack([y_train, poisoned_label])
    poisoned_model = SVC(kernel='linear')
    poisoned_model.fit(x_poisoned, y_poisoned)
    GUI.show_SVM(model, x_train, y_train, poisoned_model, x_poisoned, y_poisoned)
