# -*- coding: utf-8 -*-
import dataiku
import pandas as pd, numpy as np
import pandas as pd
import numpy as np
import math
import os
from datetime import datetime as dt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from collections import Counter
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc)
from sklearn.metrics import (r2_score, explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error)
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
from dateutil import parser
from dateutil.relativedelta import relativedelta
import joblib
import xgboost as xgb
from datetime import datetime as dt

def get_x_test(df_features, class_labels):

    # create test X and Y
    X_test = df_features.copy()
    X_test = X_test.drop(class_labels, axis=1)
    X_test = X_test.drop(["access_start_date", "tenure_length_capped", "is_churn"], axis=1)

    return X_test

def load_most_recent_model(folder_id):
    # Path to where models are saved
    path_to_models = '/home/dataiku/dss/managed_datasets/c4vR0l/{}/'.format(folder_id)

    # Get names of all models
    all_models = sorted(os.listdir(path_to_models), reverse=True)
    # Get name of most recent model
    most_recent_model = all_models[0]
    # Load model
    rfc_models = joblib.load(os.path.join(path_to_models, most_recent_model))
    return rfc_models

def get_final_prediction_from_probabilities(df):

    pred_list = ['pred_proba_target_is_3M',
                  'pred_proba_target_is_4M',
                  'pred_proba_target_is_5M',
                  'pred_proba_target_is_6M',
                  'pred_proba_target_is_7M',
                  'pred_proba_target_is_8M',
                  'pred_proba_target_is_9M',
                  'pred_proba_target_is_10M',
                  'pred_proba_target_is_11M',
                  'pred_proba_target_is_12M_plus']

    df_preds = df.copy()
    df_preds = df_preds.drop('tenure_month', axis=1)

    max = df_preds.idxmax(axis=1)

    max_new = []
    for i in max:
        val = pred_list.index(i) + 3
        max_new.append(val)

    df_preds['predicted'] = max_new

    return df_preds
