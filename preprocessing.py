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

def _payment_type_mapping(var_value):
    if var_value in ['CreditCard', 'CreditCardReferenceTransaction']:
        return 'Credit Card'
    if var_value in ['BankTransfer', 'Bank Transfer']:
        return 'Bank Transfer'
    if var_value in ['Amazon', 'Amazon Pay']:
        return 'Amazon'
    if var_value in ['Apple', 'Apple Pay']:
        return 'Apple'
    else:
        return var_value

def prepare_feature_payment_type(df):
    payment_method_unfiltered = ['Apple', 'Apple Pay', 'Credit Card', 'CreditCard', 'CreditCardReferenceTransaction', 'BankTransfer', 'Bank Transfer', 'PayPal', 'Amazon', 'Amazon Pay', 'Direct Debit']
    payment_method_feature_list = ['Apple', 'Credit Card', 'Bank Transfer', 'PayPal', 'Amazon', 'Direct Debit']

    df['payment_method'] = df['payment_method'].apply(lambda x: _payment_type_mapping(x) if x in payment_method_unfiltered else np.nan)
    return payment_method_feature_list

def create_binary_targets(df, start_month):

    # create new binary targets, one for each possible class
    start_month = start_month - 1

    class_labels = []

    m12_label = 'is_12M_plus'
    df[m12_label] = df['tenure_length_capped'].apply(lambda x: 1 if x == 12 else 0)

    for target_month in range(11, start_month, -1):
        temp_month_label = 'is_{}M'.format(target_month)
        df[temp_month_label] = df['tenure_length_capped'].apply(lambda x: 1 if x == target_month else 0)

        class_labels.append(temp_month_label)

    all_class_labels = [m12_label] + class_labels
    return df, all_class_labels

def imputate_nans(df, vars_to_impute, imputations_dict):

    imputations_list = list(imputations_dict.keys())
    if not set(vars_to_impute).issubset(imputations_list):
        raise ValueError('Please specify an imputation in imputations_dict for all variables in vars_to_impute.')


    for var in vars_to_impute:
        if var == 'months_to_competition_end':
            df["months_to_competition_end"] = df["months_to_competition_end"].apply(lambda x: imputations_dict['months_to_competition_end'] if (math.isnan(x)) | (x > 12) else x)

        else:
            df[var] = df[var].fillna(imputations_dict[var])


def cyclical_feature_encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data

def append_dummies_list(categoricals, shortlists):
    dummy_list = []
    for cat_num, categorical in enumerate(categoricals):
        for index, i in enumerate(categorical):
            for j in shortlists[cat_num]:
                dummy_list.append('{}_{}'.format(i,j))
    return dummy_list


def preprocessing_function(df,
                           current_tm_list,
                           target_month_list,
                           preferred_sports_shortlist,
                           preferred_competition_shortlist,
                           acquisition_cohort_shortlist,
                           preferred_competition_dist_shortlist,
                           preferred_sport_dist_shortlist,
                           payment_method_shortlist):

    # Unify payment types (e.g. reference CreditCard and Credit Card both as Credit Card)
    payment_method_feature_list = prepare_feature_payment_type(df)

    # Preprocessing features using the specified values as per the shortlists
    df['preferred_sport_by_hours'] = df['preferred_sport_by_hours'].apply(lambda x: x if x in preferred_sports_shortlist else 'Other')
    df['preferred_sport_by_hours_in_trip'] = df['preferred_sport_by_hours_in_trip'].apply(lambda x: x if x in preferred_sports_shortlist else 'Other')
    df['preferred_competition_by_hours'] = df['preferred_competition_by_hours'].apply(lambda x: x if x in preferred_competition_shortlist else 'Other')
    df['preferred_competition_1_in_trip_by_hours'] = df['preferred_competition_1_in_trip_by_hours'].apply(lambda x: x if x in preferred_competition_shortlist else 'Other')
    df['preferred_competition_2_in_trip_by_hours'] = df['preferred_competition_2_in_trip_by_hours'].apply(lambda x: x if x in preferred_competition_shortlist else 'Other')
    df['acq_cohort_l3'] = df['acq_cohort_l3'].apply(lambda x: x if x in acquisition_cohort_shortlist else 'Other')
    df['shared_account_proxy_binary'] = df['shared_account_proxy'].apply(lambda x: x if x > 0 else 0)
    df['previous_churn_binary'] = df['count_previous_churn'].apply(lambda x: 1 if x > 0 else 0)


    # Defining which categorical features we will dummy encode
    sport_categoricals =    [u'preferred_sport_by_hours',u'preferred_sport_by_hours_in_trip']
    sport_dist_categoricals = [u'preferred_sport_in_trip_distribution_desc']
    competition_categoricals = ['preferred_competition_by_hours', u'preferred_competition_1_in_trip_by_hours', u'preferred_competition_2_in_trip_by_hours']
    competition_dist_categoricals = [u'preferred_competition_in_trip_distribution_desc']
    acq_cohort_categoricals = [u'acq_cohort_l3']
    payment_method_categoricals = ['payment_method']

    categoricals_to_dummy_encode = payment_method_categoricals + acq_cohort_categoricals + sport_categoricals + sport_dist_categoricals + competition_categoricals + competition_dist_categoricals
    categoricals = [sport_categoricals, sport_dist_categoricals, competition_categoricals, competition_dist_categoricals, acq_cohort_categoricals, payment_method_categoricals]
    shortlists = [preferred_sports_shortlist, preferred_sport_dist_shortlist, preferred_competition_shortlist, preferred_competition_dist_shortlist, acquisition_cohort_shortlist, payment_method_shortlist]

    # create a list of the dummies expected to match with data trained on
    dummies_with_prefix_list = []
    for cat_num, categorical in enumerate(categoricals):
        for index, i in enumerate(categorical):
            for j in shortlists[cat_num]:
                dummies_with_prefix_list.append('{}_{}'.format(i,j))

    ## Create dummies
    created_dummy_feature_list = []
    for var in categoricals_to_dummy_encode:
        df_temp_dummy =  pd.get_dummies(df[var], prefix=var)
        df_temp_dummy.reset_index()
        for i in df_temp_dummy.columns.values:
            created_dummy_feature_list.append(i)
        df = df.drop(var, axis=1)
        df = pd.concat([df, df_temp_dummy], axis=1)

    # Logic for when not all dummies trained on are available in the predict sample
    for i in dummies_with_prefix_list:
        if(i not in created_dummy_feature_list):
            df[i] = 0
            created_dummy_feature_list.append(i)

    ## Now need to impute NaNs
    all_columns = df.isna().any()
    isnan_list = np.array(all_columns)
    vars_to_impute = [all_columns.index[i] for i, x in enumerate(isnan_list) if x]

    ## Create a dictionary mapping each column with the appropriate nan replacement
    imputations_dict = {}
    for i in vars_to_impute:
        imputations_dict[i] = 0

    # Impute the nans
    imputate_nans(df, vars_to_impute, imputations_dict)

    ## encode months to cyclical features - lmao does this even make sense?!
    cyclical_month_features_to_encode = ['access_start_calendar_month']
    for var in cyclical_month_features_to_encode:
        cyclical_feature_encode(df, var, 12)

    # Select model features to use
    feature_list = [
        'tenure_month',
        'access_start_calendar_month_sin',
        'access_start_calendar_month_cos',
        'ft_with_gc',
        'has_sc',
        'sleeping_baby',
        'shared_account_proxy_binary',
        'previous_churn_binary',
        'distinct_lr_devices_count',
        'proportion_of_streams_living_room',
        'number_of_device_hardware_types',
        'pper',
        'pper_in_trip',
        'had_good_quality_in_trip',
        'number_of_competitions_60min_threshold',
        'months_to_competition_end',
        'preferred_competition_1_in_trip_months_to_competition_end',
        'preferred_competition_2_in_trip_months_to_competition_end',
        'playing_time_hrs',
        'number_of_streams',
        'perc_play_time_change_first_vs_last_two_weeks',
        'playing_time_vs_avg',
        'count_distinct_stream_sessions_vs_avg'
    ]

    feature_list = feature_list + created_dummy_feature_list

    model_diagnostics={}
    for i, current_tm in enumerate(current_tm_list):
        for k, target_month in enumerate(target_month_list):
            if k >= i: model_diagnostics['tm_{}_target_{}'.format(current_tm, target_month)] = {}

    ## create binary target
    df, class_labels = create_binary_targets(df, start_month = 3)

    required_train_set_columns = (['access_start_date', 'tenure_length_capped', 'is_churn']
                 + feature_list
                 + class_labels
                )

    df_features = df.loc[:,required_train_set_columns].copy()

    return df_features, model_diagnostics, class_labels
