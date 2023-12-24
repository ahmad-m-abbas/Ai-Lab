import math
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tabulate import tabulate
from sklearnex import patch_sklearn 
patch_sklearn()
from sklearn.utils import shuffle
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from skimage import feature as ft
import time
import gc
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
class Utils:
    def find_best_rf_parameters(X_train, y_train):
        rf_param_grid = {
            'n_estimators': [10, 50, 100],
            'criterion': ['gini', 'entropy'],
            'max_depth': [10, 50, 100],
            'max_features': ["sqrt", "log2"]
        }
    
        random_search = RandomizedSearchCV(RandomForestClassifier(), rf_param_grid, n_iter=10, cv=3, verbose=2, n_jobs=-1)
        
        start_time = time.time()
        random_search.fit(X_train, y_train)
    
        print("\nAll Results:")
        for mean_score, params in zip(random_search.cv_results_['mean_test_score'], random_search.cv_results_['params']):
            print(f"{params} with accuracy: {mean_score}")
    
        # Find the best model
        best_model = random_search.best_estimator_
        print("\n--- %s seconds ---" % (time.time() - start_time))
        print("The highest Accuracy {:.5f} is the model with parameters: {}".format(random_search.best_score_, random_search.best_params_))
    
        return best_model

    def accuracy_measure_rf(X_train, y_train, X_test, y_test, n_estimators = 50, criterion  = 'entropy', max_depth =10, max_features='log2'):
        start_time = time.time()
        rf_model = RandomForestClassifier(n_estimators = n_estimators, criterion  = criterion, max_depth =max_depth, max_features=max_features)
        rf_model.fit(X_train, y_train)
        rf_accuracy=rf_model.score(X_test, y_test)
        predictions = rf_model.predict(X_test)
        rf_accuracy= accuracy_score(y_test, predictions)
        rf_f_score = precision_recall_fscore_support(y_test,predictions)
        print("--- %s seconds ---" % (time.time() - start_time))
        return rf_accuracy, rf_f_score

    def find_best_xgb_parameters(X_train, y_train, X_val, y_val):
        xgb_param_grid = {
            'n_estimators': [10, 50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [5, 50, 100]
        }
        start_time = time.time()
        xgb_model = xgb.XGBClassifier(enable_categorical=True, early_stopping_rounds=10, n_jobs=-1)
        random_search = RandomizedSearchCV(xgb_model, xgb_param_grid, n_iter=10, cv=3, verbose=2, n_jobs=2)
        random_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        print("--- %s seconds ---" % (time.time() - start_time))
        # Log all results with their accuracies
        print("All Results:")
        for mean_score, params in zip(random_search.cv_results_['mean_test_score'], random_search.cv_results_['params']):
            print(f"{params} with accuracy: {mean_score}")
    
        # Best Model Information
        best_model = random_search.best_estimator_
        print("\nBest parameters found: ", random_search.best_params_)
        print("Best accuracy found: ", random_search.best_score_)
    
        return best_model
    def accuracy_measure_xgb(X_train, y_train, X_test, y_test, n_estimators = 50, learning_rate  = 0.1, max_depth =10, subsample=0.5, colsample_bytree=0.5):
        start_time = time.time()
        xgb_model = xgb.XGBClassifier(n_estimators = n_estimators, learning_rate  = learning_rate, max_depth =max_depth, subsample=subsample, colsample_bytree=colsample_bytree)
        xgb_model.fit(X_train, y_train)
        xgb_accuracy=xgb_model.score(X_test, y_test)
        predictions = xgb_model.predict(X_test)
        xgb_accuracy= accuracy_score(y_test, predictions)
        xgb_f_score = precision_recall_fscore_support(y_test,predictions)
        print("--- %s seconds ---" % (time.time() - start_time))
        return xgb_accuracy, xgb_f_score
    