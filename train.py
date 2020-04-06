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
    payment_method_unfiltered = ['Apple', 'Apple Pay', 'Credit Card', 'CreditCard', 'CreditCardReferenceTransaction', 'BankTransfer', 'Bank Transfer', 'PayPal', 'Amazon', 'Amazon Pay', 'Direct Debit', 'Google Play Billing']
    payment_method_feature_list = ['Apple', 'Credit Card', 'Bank Transfer', 'PayPal', 'Amazon', 'Direct Debit', 'Google Play Billing']

    df['payment_method'] = df['payment_method'].apply(lambda x: _payment_type_mapping(x) if x in payment_method_unfiltered else np.nan)
    return payment_method_feature_list

def _cohort_type_mapping(var_value):
    if var_value in ['Google', 'Google Tracking ID']:
        return 'Google'
    else:
        return var_value

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
                           payment_method_shortlist,
                           alt_market)

    # Unify payment types (e.g. reference CreditCard and Credit Card both as Credit Card)
    payment_method_feature_list = prepare_feature_payment_type(df)

    # Preprocessing features using the specified values as per the shortlists
    df['preferred_sport_by_hours'] = df['preferred_sport_by_hours'].apply(lambda x: x if x in preferred_sports_shortlist else 'Other')
    df['preferred_sport_by_hours_in_trip'] = df['preferred_sport_by_hours_in_trip'].apply(lambda x: x if x in preferred_sports_shortlist else 'Other')
    df['preferred_competition_by_hours'] = df['preferred_competition_by_hours'].apply(lambda x: x if x in preferred_competition_shortlist else 'Other')
    df['preferred_competition_1_in_trip_by_hours'] = df['preferred_competition_1_in_trip_by_hours'].apply(lambda x: x if x in preferred_competition_shortlist else 'Other')
    df['preferred_competition_2_in_trip_by_hours'] = df['preferred_competition_2_in_trip_by_hours'].apply(lambda x: x if x in preferred_competition_shortlist else 'Other')

    #Removed due to snowflake migration
    #df['acq_cohort_l3'] = df['acq_cohort_l3'].apply(lambda x: _cohort_type_mapping(x))
    #df['acq_cohort_l3'] = df['acq_cohort_l3'].apply(lambda x: x if x in acquisition_cohort_shortlist else 'Other')
    df['shared_account_proxy_binary'] = df['shared_account_proxy'].apply(lambda x: x if x > 0 else 0)
    df['previous_churn_binary'] = df['count_previous_churn'].apply(lambda x: 1 if x > 0 else 0)
    df['payment_method'] = df['payment_method'].apply(lambda x: x if x in payment_method_shortlist else 'Other')


    # Defining which categorical features we will dummy encode
    sport_categoricals =    [u'preferred_sport_by_hours',u'preferred_sport_by_hours_in_trip']
    sport_dist_categoricals = [u'preferred_sport_in_trip_distribution_desc']
    competition_categoricals = ['preferred_competition_by_hours', u'preferred_competition_1_in_trip_by_hours', u'preferred_competition_2_in_trip_by_hours']
    competition_dist_categoricals = [u'preferred_competition_in_trip_distribution_desc']
    #acq_cohort_categoricals = [u'acq_cohort_l3']
    payment_method_categoricals = ['payment_method']

    categoricals_to_dummy_encode = payment_method_categoricals + sport_categoricals + sport_dist_categoricals + competition_categoricals + competition_dist_categoricals
    #+ acq_cohort_categoricals


    categoricals = [
        sport_categoricals,
        sport_dist_categoricals,
        competition_categoricals,
        competition_dist_categoricals,
        #acq_cohort_categoricals,
         payment_method_categoricals]

    shortlists = [
         preferred_sports_shortlist,
         preferred_sport_dist_shortlist,
         preferred_competition_shortlist,
         preferred_competition_dist_shortlist,
         #acquisition_cohort_shortlist,
         payment_method_shortlist]

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

    ## encode months to cyclical features - plz make sense?!
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
        'distinct_LR_devices_count',
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

    if alt_market:
        model_diagnostics={}
        for i, current_tm in enumerate(current_tm_list):
            for k, target_month in enumerate(target_month_list):
                if current_tm >= i+3: model_diagnostics['tm_{}_target_{}'.format(current_tm, target_month)] = {}

        ## create binary target
        df, class_labels = create_binary_targets(df, start_month = 3)

        required_train_set_columns = (['access_start_date', 'tenure_length_capped', 'is_churn']
                     + feature_list
                     + class_labels
                    )

        df_features = df.loc[:,required_train_set_columns].copy()

        # Create a list containing all the global models that aren't a part of the local model list
        class_labels_to_drop = [i for i in class_labels if i not in target_month_list]
        print('dropping the following target month cols: {}'.format(class_labels_to_drop))

        df_features = df_features.drop(class_labels_to_drop, axis=1)
    else:
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


def get_x_and_y_train(current_tm_list,
                      target_month_list,
                      df_features_train,
                      synchronization_time_days,
                      model_diagnostics):

    # Create empty dict for X and Y train sets per model
    X_train_dict = {}
    Y_train_dict = {}

    # Bring in the training calculation dates from the DSS project variables
    train_date = dataiku.get_custom_variables()['training_calculation_date']
    training_start_date = dataiku.get_custom_variables()['training_start_date']

    # For each current tenure month
    for i, current_tm in enumerate(current_tm_list):

        # For each target month per current tenure month
        for k, target_month in enumerate(target_month_list):
            # As long as the target month is greater or equal to the current month
            if k >= i:

                target_month_num = 2 + k
                print('building training set for current tm {} and target month {}'.format(current_tm, target_month))
                current_tm_df = df_features_train[df_features_train['tenure_month'] == current_tm - 1]

                # create end and start period for training
                training_end_date = parser.parse(train_date) + relativedelta(months=-target_month_num, days = - synchronization_time_days)

                # convert datetime to string
                training_start_date_str = training_start_date
                training_end_date_str = training_end_date.strftime('%Y-%m-%d')

                # Create mask filters based on the calculation date and apsply using the access start date field
                start_mask = current_tm_df['access_start_date'] >= training_start_date_str
                end_mask = current_tm_df['access_start_date'] <= training_end_date_str
                current_tm_df = current_tm_df.loc[start_mask & end_mask]

                # Define the class labels to drop
                print("initial train sample size     : " + str(current_tm_df.shape[0]))
                class_labels_to_drop = list(target_month_list)
                class_labels_to_drop.remove(target_month)
                print("class_labels_to_drop: {}".format(class_labels_to_drop))

                # Downsample to have a balanced split of the target
                class_weight = current_tm_df[target_month].sum() / 1.0 / current_tm_df.shape[0]
                print("class weight: " + str(class_weight))
                sample_size = current_tm_df[target_month].sum()
                df_features_train_sampled = current_tm_df.groupby(target_month, group_keys=False).apply(lambda group: group.sample(sample_size, replace=True))

                # Output to console
                total_train = df_features_train_sampled.shape[0]
                percentage_target_train = df_features_train_sampled[target_month].sum() / float(total_train)
                print("downsampled train sample size     : " + str(total_train))
                print("% target class        : " + str(percentage_target_train))
                print('tm_{}_target_{}'.format(current_tm,target_month) + ': train_samples: ' + str(df_features_train_sampled.shape[0]) + '; start_time:', df_features_train_sampled['access_start_date'].min().strftime('%Y-%m-%d'), '; end_time:', df_features_train_sampled['access_start_date'].max().strftime('%Y-%m-%d'))

                # Output to the model diagnostics dict
                model_diagnostics['tm_{}_target_{}'.format(current_tm,target_month)]['training_samples_available'] = df_features_train_sampled.shape[0]
                model_diagnostics['tm_{}_target_{}'.format(current_tm,target_month)]['train_start_date'] = df_features_train_sampled['access_start_date'].min().strftime('%Y-%m-%d')
                model_diagnostics['tm_{}_target_{}'.format(current_tm,target_month)]['train_end_date'] = df_features_train_sampled['access_start_date'].max().strftime('%Y-%m-%d')

                # Create train X and Y
                X_train = df_features_train_sampled.copy()
                X_train = X_train.drop(class_labels_to_drop, axis=1)
                X_train = X_train.drop(target_month, axis=1)
                X_train = X_train.drop(["access_start_date", "tenure_length_capped", "is_churn"], axis=1)
                Y_train = df_features_train_sampled.loc[:,[target_month]].values.ravel()

                # Add the X and Y train sets to the dict for all models
                X_train_dict['tm_{}_target_{}'.format(current_tm,target_month)] = X_train
                Y_train_dict['tm_{}_target_{}'.format(current_tm,target_month)]  = Y_train

    return X_train_dict, Y_train_dict, model_diagnostics


def train_models(current_tm_list, target_month_list, X_train_dict, Y_train_dict, model_diagnostics):

    model_list = []
    rfc_models = {}

    for i, current_tm in enumerate(current_tm_list):

        for k, target_month in enumerate(target_month_list):
            if k >= i:

                # train model
                start = dt.now()
                print("training tm {} target {} model".format(current_tm,target_month))

                rfc = RandomForestClassifier(oob_score=True,
                             random_state = 0,
                             n_estimators=200,
                             min_samples_split=0.01,
                             max_depth=6,
                             #min_samples_leaf=0.01,
                             #class_weight = 'balanced',
                             #max_features = 0.3,
                            )

                rfc_model = rfc.fit(X_train_dict['tm_{}_target_{}'.format(current_tm, target_month)], Y_train_dict['tm_{}_target_{}'.format(current_tm, target_month)])

                print("training tm {} target {} model took: {}".format(current_tm, target_month, dt.now()-start))
                print("trained tm {} target {} model with oob score: ".format(current_tm, target_month) + str(rfc_model.oob_score_))

                model_list.append(rfc_model)
                rfc_models['tm_{}_target_{}'.format(current_tm, target_month)] = rfc_model

                # Output to console
                total_train = Y_train_dict['tm_{}_target_{}'.format(current_tm, target_month)].shape[0]
                total_target_class = Y_train_dict['tm_{}_target_{}'.format(current_tm, target_month)].sum()
                class_weight = total_target_class / float(total_train)

                model_diagnostics['tm_{}_target_{}'.format(current_tm, target_month)]['oob_score'] = rfc_model.oob_score_
                model_diagnostics['tm_{}_target_{}'.format(current_tm, target_month)]['model_parameter'] = rfc

                model_diagnostics['tm_{}_target_{}'.format(current_tm, target_month)]['total_train_samples_used'] = total_train
                model_diagnostics['tm_{}_target_{}'.format(current_tm, target_month)]['total_train_samples_target_class'] = total_target_class
                model_diagnostics['tm_{}_target_{}'.format(current_tm, target_month)]['total_train_percentage_target_class'] = class_weight
                model_diagnostics['tm_{}_target_{}'.format(current_tm, target_month)]['sampling_strategy'] = 'No sampling strategy'

                train_preds = rfc_model.predict(X_train_dict['tm_{}_target_{}'.format(current_tm, target_month)])
                model_diagnostics['tm_{}_target_{}'.format(current_tm, target_month)]['train_set_accuracy_score'] = accuracy_score(Y_train_dict['tm_{}_target_{}'.format(current_tm, target_month)], train_preds)
                model_diagnostics['tm_{}_target_{}'.format(current_tm, target_month)]['train_set_precision_score'] = precision_score(Y_train_dict['tm_{}_target_{}'.format(current_tm, target_month)], train_preds)
                model_diagnostics['tm_{}_target_{}'.format(current_tm, target_month)]['train_set_recall_score'] = recall_score(Y_train_dict['tm_{}_target_{}'.format(current_tm, target_month)], train_preds)

    return rfc_models, model_diagnostics


def save_models(rfc_models, dss_folder_id, model_name):

    models = dataiku.Folder(dss_folder_id)
    path_to_folder = models.get_path()
    current_time = dt.today().strftime('%Y-%m-%d_%H-%M-%S')
    path_to_save_model = os.path.join(path_to_folder, current_time + '_' + model_name)

    path_to_save_compressed_model = path_to_save_model + '_compressed_v01'
    model_save_path = joblib.dump(rfc_models, path_to_save_compressed_model, compress = True)
    print(model_save_path)

def save_model_diagnostics(current_tm_list, target_month_list, model_diagnostics, X_train_dict, rfc_models, output_table_name):

    for i, current_tm in enumerate(current_tm_list):
        for k, target_month in enumerate(target_month_list):
            if k >= i:
                feature_importance_dict = dict(zip(X_train_dict['tm_{}_target_{}'.format(current_tm,target_month)].columns, rfc_models['tm_{}_target_{}'.format(current_tm,target_month)].feature_importances_))
                model_diagnostics['tm_{}_target_{}'.format(current_tm,target_month)]['feature_importance'] = feature_importance_dict

    diagnostics_df = pd.DataFrame(model_diagnostics).transpose()
    diagnostics_df = diagnostics_df.dropna()
    diagnostics_df['model_training_date'] = dt.today().strftime('%Y-%m-%d %H:%M')
    diagnostics_df['model'] = diagnostics_df.index
    diagnostics_df['training_calculation_date'] = dataiku.get_custom_variables()['training_calculation_date']

    cols = ['model', 'model_training_date', 'training_calculation_date', 'train_start_date', 'train_end_date', 'training_samples_available', 'total_train_percentage_target_class', 'total_train_samples_target_class', 'total_train_samples_used', 'oob_score', 'model_parameter', 'feature_importance']
    diagnostics_df = diagnostics_df[cols]

    train_model_diagnostics = dataiku.Dataset(output_table_name)
    train_model_diagnostics.write_with_schema(diagnostics_df)


def calculate_average_docomo_tenure_length(df_docomo, synchronization_time_days):

    one_year_ago = parser.parse(dataiku.get_custom_variables()['training_calculation_date']) + relativedelta(months=-12, days = -synchronization_time_days)
    two_years_ago = parser.parse(dataiku.get_custom_variables()['training_calculation_date']) + relativedelta(months=-24, days = -synchronization_time_days)
    df_docomo_time_filter_max = df_docomo['access_start_date'] < one_year_ago
    df_docomo_time_filter_min = df_docomo['access_start_date'] > two_years_ago

    df_docomo_filtered = df_docomo[df_docomo_time_filter_max & df_docomo_time_filter_min].copy()
    if df_docomo_filtered.size > 0:
        docomo_sleeping_babies_average_tenure_length = df_docomo_filtered['tenure_length_capped'].mean()
    else:
        docomo_sleeping_babies_average_tenure_length = 12.0

    return docomo_sleeping_babies_average_tenure_length

def get_regr_models(df_out, current_tm_list):

    regr_max_date = (parser.parse(dataiku.get_custom_variables()['training_calculation_date']) +  relativedelta(months=-11)).strftime('%Y-%m-%d')
    regr_df = df_out[df_out['access_start_date'] <= (dataiku.get_custom_variables()['training_calculation_date']) ]

    X_regr_dict = {}
    Y_regr_dict = {}

    for i in current_tm_list:
            current_tm_df = regr_df[regr_df['tenure_month'] == i-1]
            Y_regr_dict['tm_{}'.format(i)] = current_tm_df['tenure_length_capped']
            X_regr_dict['tm_{}'.format(i)] = current_tm_df.copy()
            X_regr_dict['tm_{}'.format(i)] = X_regr_dict['tm_{}'.format(i)].drop(class_labels, axis=1)
            X_regr_dict['tm_{}'.format(i)] = X_regr_dict['tm_{}'.format(i)].drop(['access_start_date', 'tenure_length_capped', 'is_churn'], axis=1)

    xgb_model_regr_dict = {}
    xgb_model_regr_labels = X_regr_dict['tm_3'].columns.values
    y_labels_dict = Y_regr_dict.copy()

    for i, current_tm in enumerate(X_regr_dict.keys()):
        start = dt.now()
        print("training xgb regressor for users in current month {}".format(current_tm))
        clf = xgb.XGBRegressor(objective ='reg:squarederror',
                              learning_rate = 0.1,
                              max_depth = 7,
                              gamma = 0,
                              reg_lambda = 1,
                              n_estimators = 25,
                              )

        clf = clf.fit(X_regr_dict[current_tm], y_labels_dict[current_tm])
        xgb_model_regr_dict[current_tm] = clf
        print('training took: {}'.format(dt.now()-start))

    return xgb_model_regr_dict, xgb_model_regr_labels
