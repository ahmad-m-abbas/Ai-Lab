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

class Utils:
    def find_best_rf_parameters(X_train, y_train, X_val, y_val):   
        rf_param_grid = {
            'n_estimators': [10, 50, 100],
            'criterion': ['entropy', 'gini'],
            'max_depth': [10, 50, 100],
            'max_features': ["sqrt", "log2"]
        }
        rf_grid = ParameterGrid(rf_param_grid)
        data = []
        head = ['n_estimators', 'criterion', 'max_depth', 'max_features', 'score in validation set']
    
        start_time = time.time()
        for param in rf_grid:
            rf_model = RandomForestClassifier(**param)
            rf_model.fit(X_train, y_train)
            
            y_pred = rf_model.predict(X_val)
            score = accuracy_score(y_val, y_pred)
            list_entry = [param['n_estimators'],  param['criterion'], param['max_depth'], param['max_features'], score]
            data.insert(0, list_entry)
            del rf_model, y_pred
            gc.collect()
    
        # Find the best model
        max_accuracy = max(entry[4] for entry in data)
        best_model = max(data, key=lambda x: x[4])
    
        # Print the results in a table format
        print(tabulate(data, headers=head, tablefmt="pipe"))
        print("--- %s seconds ---" % (time.time() - start_time))
        print("The highest Accuracy {:.5f} is the model with n_estimators = {}, criterion = '{}', max_depth = {}, and max_features = '{}'"
              .format(best_model[4], best_model[0], best_model[1], best_model[2], best_model[3]))
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
            'n_estimators': [10, 100],
            'learning_rate': [0.01, 0.1, 0.5],
            'max_depth': [10, 50]
        }
        xgb_grid = ParameterGrid(xgb_param_grid)
        data = []
        head = ['n_estimators', 'learning_rate', 'max_depth', 'score in validation set']
    
        start_time = time.time()
        for param in xgb_grid:
            xgb_model = xgb.XGBClassifier(**param)
            xgb_model.fit(X_train, y_train)
            y_pred = xgb_model.predict(X_val)
            score = accuracy_score(y_val, y_pred)
            list_entry = [param['n_estimators'], param['learning_rate'], param['max_depth'], score]
            data.insert(0, list_entry)
            del xgb_model, y_pred
            gc.collect()
            print("Finished", param)
    
        # Find the best model
        max_accuracy = max(entry[5] for entry in data)
        best_model = max(data, key=lambda x: x[5])
    
        # Print the results in a table format
        print(tabulate(data, headers=head, tablefmt="pipe"))
        print("--- %s seconds ---" % (time.time() - start_time))
        print("The highest Accuracy {:.5f} is the model with n_estimators = {}, learning_rate = {}, max_depth = {}"
              .format(best_model[5], best_model[0], best_model[1], best_model[2]))
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
    