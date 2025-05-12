from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np


X_data = ...
t_data = ...


###########################
# MODEL FITTING AND EVALUATION
###########################

def fit_model(model, X_data, t_data, cross_val=None):
    
    """Fits model the X_data and t_data
    if cross_val is not None we use cross validation to train and evaluate the model with cross_val folds

    :param model: One of the models listed below
    :type model: Any
    :param X_data: Training data points
    :type X_val: np.array
    :param t_val: Training data labels
    :type t_val: np.array
    :param cross_val: Indicates whether we should use cross validation or not
    :type cross_val: int, defaults to None
    :return: Return the fitted model and the accuracy on the either the training set and validation set, 
    or the average accuracy and standard deviation for all of the folds in the case of cross validation
    :rtype: _type_   
    """
    
    if cross_val is None:
        X_train, X_val, t_train, t_val = train_test_split(X_data, t_data, test_size=0.2)
        model.fit(X_train, t_train)
        train_acc = model.score(X_train, t_train)
        val_acc = model.score(X_val, t_val)
        return model, train_acc, val_acc
    else:
        scores = cross_val_score(model, X_data, t_data, cv=cross_val, scoring='neg_mean_squared_error')
        mean_rmse = np.sqrt(-scores.mean())
        return model, mean_rmse


###########################
# MODEL TRAINING
###########################


def generate_decision_tree(max_depth) -> DecisionTreeRegressor:
    """Generates a DecisionTreeRegressor

    :return: Descion tree fitted to the training data
    :rtype: DecisionTreeRegressor
    """
    tree = DecisionTreeRegressor(max_depth=max_depth)
    return tree

def generate_random_forest(num_estimators, max_depth) -> RandomForestRegressor:
    """Generates a RandomForestRegressor

    :param num_estimators: Number of descion trees to train
    :type num_estimators: int
    :param max_depth: Max depth of the trees
    :type max_depth: int
    :return: Random forest fitted to the training data
    :rtype: RandomForestRegressor
    """
    forest = RandomForestRegressor(n_estimators=num_estimators, max_depth=max_depth)
    return forest

def generate_gradient_boosted_trees(num_estimators, max_depth, learning_rate) -> GradientBoostingRegressor:
    """generates a GradientBoostingRegressor
    
    :param num_estimators: Number of descion trees to train
    :type num_estimators: int
    :param max_depth: Max depth of the trees
    :type max_depth: int
    :param learning_rate: Learning rate at which to train
    :type learning_rate: float
    :return: Returns a gradient boosted regressor
    :rtype: GradientBoostingRegressor
    """
    
    boost = GradientBoostingRegressor(n_estimators=num_estimators, max_depth=max_depth, learning_rate=learning_rate)
    return boost

def generate_knn(num_neighbors) -> KNeighborsRegressor:
    """Generates a KNeighborsRegressor model

    :return: KNN model
    :rtype: KNeighborsRegressor
    """
    knn = KNeighborsRegressor(n_neighbors=num_neighbors)
    return knn

def generate_gp() -> GaussianProcessRegressor:

    """Generates a GaussianProcessRegressor
    :return: Returns a gaussian process
    :rtype: GaussianProcessRegressor
    """

    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    return gaussian_process