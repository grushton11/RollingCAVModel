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

def get_rfc_models_dict(current_tm_list, target_month_list, rfc_models):
    rfc_models_dict = {}
    for i in current_tm_list:
        in_tm_dict = {}
        for target_index, j in enumerate(target_month_list):
            if target_index+3 >= i:
                in_tm_dict['target_{}'.format(j)] = rfc_models['tm_{}_target_{}'.format(i, j)]
        rfc_models_dict['tm_{}'.format(i)] = in_tm_dict
    return rfc_models_dict

def get_predictions_df_dict(X_test_dict, rfc_models_dict):
    predictions_df_dict = {}

    for current_month in X_test_dict.keys():
        predictions_df_dict[current_month] = X_test_dict[current_month].copy()
        predictions_final_df = pd.DataFrame([])
        predictions_final_df['tenure_month'] = predictions_df_dict[current_month]['tenure_month']

        for i in rfc_models_dict[current_month].keys():
            if X_test_dict[current_month].size > 0:
                predictions = [x[1] for x in rfc_models_dict[current_month][i].predict_proba(X_test_dict[current_month])]
            else:
                predictions = []
            predictions_df_dict[current_month]['pred_proba_' + i] = predictions
            predictions_final_df['pred_proba_' + i] = predictions
            predictions_df_dict[current_month]['pred_proba_' + i] = predictions

        final_prediction = get_final_prediction_from_probabilities(predictions_final_df)
        predictions_df_dict[current_month]['prediction'] = final_prediction['predicted']

    return predictions_df_dict

def create_results_dict(predictions_df_dict):
    df_rfc_results_dict = {}
    for current_month in predictions_df_dict.keys():
        df_rfc_results_dict[current_month] = pd.DataFrame({'tenure_months_completed' : predictions_df_dict[current_month]['tenure_month'],
                                                           'current_tenure_month' : predictions_df_dict[current_month]['tenure_month'] + 1,
                                                           'prediction': predictions_df_dict[current_month]['prediction']
                                                          })
    return df_rfc_results_dict


def add_prediction_identifiers(df_rfc_results_dict, test_dict):

    identifier_columns = ['cust_account_id','cust_territory', 'cust_country','access_start_date']
    prediction_columns = ['tenure_months_completed', 'current_tenure_month', 'prediction']

    for current_month in df_rfc_results_dict:
        df_rfc_results_dict[current_month] = df_rfc_results_dict[current_month].join(test_dict[current_month])[identifier_columns + prediction_columns]
        prediction_date = dataiku.get_custom_variables()['prediction_date']
        df_rfc_results_dict[current_month]['prediction_date'] = prediction_date
        df_rfc_results_dict[current_month]['inserted_at'] = dt.today().strftime('%Y-%m-%d %H:%M')

    return df_rfc_results_dict

def get_output_file(df_rfc_results_dict):
    df_list = []
    for current_month in df_rfc_results_dict.keys():
        df_list.append(df_rfc_results_dict[current_month])

    output_df = pd.concat(df_list)
    output_df = output_df.reset_index(drop=True)

    return output_df
