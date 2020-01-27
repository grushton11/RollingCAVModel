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
                           acquisiton_cohort_shortlist,
                           preferred_competition_dist_shortlist,
                           preferred_sport_dist_shortlist,
                           payment_method_shortlist):

    # Unify payment types (e.g. reference CreditCard and Credit Card both as Credit Card)
    payment_method_feature_list = CAVmodel.prepare_feature_payment_type(df)

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
    CAVmodel.imputate_nans(df, vars_to_impute, imputations_dict)

    ## encode months to cyclical features - lmao does this even make sense?!
    cyclical_month_features_to_encode = ['tenure_month_start_calendar_month', 'access_start_calendar_month']

    for var in cyclical_month_features_to_encode:
        cyclical_feature_encode(df, var, 12)

    # Select model features to use
    feature_list = [
        'tenure_month',
        'tenure_month_start_calendar_month_sin',
        'tenure_month_start_calendar_month_cos',
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
    df, class_labels = CAVmodel.create_binary_targets(df, start_month = 3)

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
                print("current tenure month: {}, train tenure month: {}, binary target tenure month: {}".format(current_tm, current_tm-1, target_month))
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

test

def train_models(current_tm_list, target_month_list, X_train_dict, Y_train_dict, model_diagnostics, rfc):

    model_list = []
    rfc_models = {}

    for i, current_tm in enumerate(current_tm_list):

        for k, target_month in enumerate(target_month_list):
            if k >= i:

                # train model
                start = dt.now()
                print("training tm {} target {} model".format(current_tm,target_month))

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

    return rfc_models
