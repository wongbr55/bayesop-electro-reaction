from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVR

from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.model_selection import cross_val_score, train_test_split, LeaveOneOut
import numpy as np



###########################
# MODEL Training AND EVALUATION
###########################

class ModelTraining:
    
    """Methods for model training and evaluation
    """
    def __init__(self):
        pass

    def fit_model(self, model, X_data, t_data):
        
        """Fits model the X_data and t_data

        :param model: One of the models listed below
        :type model: Any
        :param X_data: Training data points
        :type X_val: np.array
        :param t_val: Training data labels
        :type t_val: np.array
        :return: Return the fitted model and the accuracy on the either the training set and validation set
        :rtype: tuple   
        """
        
        X_train, X_val, t_train, t_val = train_test_split(X_data, t_data, test_size=0.2)
        model.fit(X_train, t_train)
        train_acc = model.score(X_train, t_train)
        val_acc = model.score(X_val, t_val)
        return model, train_acc, val_acc

        
    def cross_validation(self, model, X_data, t_data, cross_val=5):
        """Trains model on X_data, t_data using cross validation with cross_val folds using negative mean sqaured error as our score

        :param model: Model from GenerateModel
        :type model: Any
        :param X_data: Training data points
        :type X_data: np.array
        :param t_data: Training data labels
        :type t_data: np.array
        :param cross_val: Number of k folds to use
        :type cross_val: int, defaults to 5
        :return: Returns the trained model and the average score of all the folds
        :rtype: tuple
        """
        
        scores = cross_val_score(model, X_data, t_data, cv=cross_val, scoring='neg_mean_squared_error')
        mean_rmse = np.sqrt(-scores.mean())
        return model, mean_rmse
    
    def leave_one_out(self, model, X_data, t_data):
        """Trains model on X_data, t_data using the leave one out method

        :param model: Model from GenerateModel
        :type model: Any
        :param X_data: Training data points
        :type X_data: np.array
        :param t_data: Training data labels
        :type t_data: np.array
        :return: Returns the trained model and the average score of all the folds
        :rtype: tuple
        """
        scores = cross_val_score(model, X_data, t_data, cv=LeaveOneOut(), scoring='neg_mean_squared_error')
        mean_rmse = np.sqrt(-scores.mean())
        return model, mean_rmse

###########################
# FEATURE SELECTION
###########################

class FeatureSelection:
    
    """Contains feature selection methods
    """
    
    def __init__(self):
        pass

    def select_from_model(self, model, X_data, pretrained):
        """Performs feature selection with SelectFromModel

        :param model: Model that the data was trained on
        :type model: Any
        :param X_data: Training data points
        :type X_data: np.array
        :param pretrained: Boolean telling us whether we pretrained the model or not
        :type pretrained: bool
        """
        
        meta_transformer = SelectFromModel(model, prefit=pretrained, threshold="mean")
        feature_info = meta_transformer.get_support()
        removed_feature_indicies = np.where(feature_info == False)[0]
        
        X_new = meta_transformer.transform(X_data)
        
        return X_new, removed_feature_indicies
        

    def variance_threshold(self, X_data, threshold):
        """Performs feature selection with the variance threshold method
        
        :param X_data: Training data points
        :type X_data: np.array
        :param threshold: Threshold for variance 
        :type: float
        :return: We return both the transformed data and the indicies of the removed features
        :rtype: _type_
        """
            
        mask = VarianceThreshold(threshold=(threshold * (1 - threshold)))
        X_new = mask.fit_transform(X_data)
        
        feature_info = mask.get_support()
        removed_feature_indicies = np.where(feature_info == False)[0]
        return X_new, removed_feature_indicies

###########################
# MODEL GENERATION
###########################

class GenerateModel:
    
    """Contains model generation methods
    """
    
    def __init__(self):
        pass

    def decision_tree(self, max_depth) -> DecisionTreeRegressor:
        """Generates a DecisionTreeRegressor

        :param max_depth: Max depth of the trees
        :type max_depth: int
        :return: Descion tree fitted to the training data
        :rtype: DecisionTreeRegressor
        """
        tree = DecisionTreeRegressor(max_depth=max_depth)
        return tree

    def random_forest(self, num_estimators, max_depth) -> RandomForestRegressor:
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

    def gradient_boosted_trees(self, num_estimators, max_depth, learning_rate) -> GradientBoostingRegressor:
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

    def knn(self, num_neighbors) -> KNeighborsRegressor:
        """Generates a KNeighborsRegressor model

        :return: KNN model
        :rtype: KNeighborsRegressor
        """
        knn = KNeighborsRegressor(n_neighbors=num_neighbors)
        return knn

    def gp(self) -> GaussianProcessRegressor:

        """Generates a GaussianProcessRegressor
        :return: Returns a gaussian process
        :rtype: GaussianProcessRegressor
        """

        kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        return gaussian_process
    
    def svr(self) -> SVR:
        """Generates an SVR

        :return: Returns an SVR
        :rtype: SVR
        """
        return SVR()
    
    def linear_regression(self) -> LinearRegression:
        """Generates a LinearRegression instance

        :return: Returns a LinearRegression
        :rtype: LinearRegression
        """
        return LinearRegression()