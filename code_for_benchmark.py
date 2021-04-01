def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Imports
import pandas as pd
import numpy as np
import random
from time import time

from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from ReliefF import ReliefF

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

from tqdm import tqdm

# Setup
target_variable = 'target'
neighbor_samples_ratio = 0.02
min_neighbors = 8
max_neighbors = 56
train_validation_split = 0.70
train_test_split = 0.70
n_iterations = 10
offset_iterations = 0

# Defining the methods
## Filter
### Eliminate high intercorrelation features
def intercorrelation_filter(train_df, threshold = 0.8):
    feature_set = train_df.drop(target_variable, axis = 1)
    corr_matrix = abs(feature_set.corr()) 
    target_corr_matrix = abs(train_df.corr())
    high_intercorrelations = corr_matrix > threshold
    features = list(corr_matrix)
    selected_features = list(corr_matrix)
    for feature_a in features:
        for feature_b in features:
            if feature_a == feature_b:
                continue
            if high_intercorrelations[feature_a][feature_b]:
                if target_corr_matrix[feature_a][target_variable] > target_corr_matrix[feature_b][target_variable]:
                    if feature_b in list(selected_features):
                        selected_features.remove(feature_b)
                else:
                    if feature_a in list(selected_features):
                        selected_features.remove(feature_a)
        features.remove(feature_a)
        
    weight_dict = {}
    for feature in list(feature_set):
        weight_dict.update({feature : abs(train_df[feature].corr(train_df[target_variable]))})

    return train_df.drop('target', axis = 1)[selected_features], selected_features, weight_dict

## Embedded
### Random forest importance
def tree_based_filter(train_df, feature_importance_threshold = 0.2):
    clf = RandomForestClassifier()
    
    feature_set = train_df.drop(target_variable, axis = 1)
    features = list(feature_set)
    clf.fit(feature_set, train_df[target_variable])
    
    weight_dict = {}
    for feature, feature_importance in zip(features, clf.feature_importances_):
        weight_dict.update({feature : feature_importance})
    for feature, feature_importance in zip(features, clf.feature_importances_):
        if feature_importance < feature_importance_threshold:
            features.remove(feature)

    return train_df[features], features, weight_dict

### L1 regularization
def l1_filter(train_df, regularization_parameter = 0.01):
    feature_set = train_df.drop(target_variable, axis = 1)

    tempclf = LinearSVC(C=regularization_parameter, penalty="l1", dual=False).fit(feature_set, train_df[[target_variable]])

    model = SelectFromModel(tempclf, prefit=True)
    features_idx = model.get_support()
    features = feature_set.columns[features_idx]
    # If no features are selected, then all features are selected
    if (features_idx == [False for _ in features_idx]).all():
        features = feature_set.columns
        #print(len(list(feature_set)), len(features), features_idx, features_idx == [False for _ in features_idx])
    return train_df[features], features

### ReliefF
def relieff(train_df, perc_features_to_keep = 0.75):

    feature_set = train_df.drop(target_variable, axis = 1)
    feature_list = list(feature_set)

    train_set = train_df.iloc[:int(train_validation_split * len(train_df))]
    test_set = train_df.iloc[int(train_validation_split * len(train_df)):]

    n_features_to_keep = int(perc_features_to_keep * len(feature_list))

    fs = ReliefF(n_neighbors=5, n_features_to_keep=n_features_to_keep)

    fs.fit(train_set[feature_list].values, train_set[target_variable].values)

    selected_features = []
    for i in range(n_features_to_keep):
        selected_features.append(feature_list[fs.top_features[i]])
    
    feature_weights = {}
    for feature, weight in zip(feature_list, fs.feature_scores):
        if weight < 0:
            feature_weights.update({feature : 1 - weight/fs.feature_scores.min()})
        else:
            feature_weights.update({feature : weight/fs.feature_scores.max()})

    return train_df[selected_features], selected_features, feature_weights

## Wrapper methods
### Forward selection
def forward_selection(clf, train_df, minimum_improvement_perc = 0.01):
    feature_set = train_df.drop(target_variable, axis = 1)
    train_set = train_df.iloc[:int(train_validation_split * len(train_df))]
    test_set = train_df.iloc[int(train_validation_split * len(train_df)):]
    selected_features = []
    
    best_performer = -9999999999

    while 1:
        results = {}
        for feature in feature_set:
            clf.fit(train_set[selected_features + [feature]], train_set[target_variable])
            results.update({feature : accuracy_score(clf.predict(test_set[selected_features + [feature]]), test_set[target_variable])})

        if (best_performer * (1 + minimum_improvement_perc)) < results[max(results, key=results.get)]:
            best_performer = results[max(results, key=results.get)]
            selected_features.append(max(results, key=results.get))
            feature_set.drop(selected_features[-1], axis = 1)
            if len(list(feature_set)) == 0:
                return train_df[selected_features], selected_features
        else:
            return train_df[selected_features], selected_features

### Backwards selection
def backwards_selection(clf, train_df, minimum_improvement_perc = 0.01):
    feature_set = train_df.drop(target_variable, axis = 1)
    train_set = train_df.iloc[:int(train_validation_split * len(train_df))]
    test_set = train_df.iloc[int(train_validation_split * len(train_df)):]
    selected_features = list(feature_set)

    best_performer = -9999999999

    while 1:
        results = {}
        for feature in selected_features:
            selected_features_test = selected_features[:]
            selected_features_test.remove(feature)
            clf.fit(train_set[selected_features_test], train_set[target_variable])
            results.update({feature : accuracy_score(clf.predict(test_set[selected_features_test]), test_set[target_variable])})

        if (best_performer * (1 + minimum_improvement_perc)) < results[max(results, key=results.get)]:
            best_performer = results[max(results, key=results.get)]
            selected_features.remove(max(results, key=results.get))
            feature_set.drop(selected_features[-1], axis = 1)
            if len(list(feature_set)) == 1:
                return train_df[selected_features], selected_features
        else:
            return train_df[selected_features], selected_features

### Stepwise selection
def stepwise_selection(clf, train_df, minimum_improvement_perc = 0.01, initial_perc = 0.5):
    remaining_feature_set = list(train_df.drop(target_variable, axis = 1))
    train_set = train_df.iloc[:int(train_validation_split * len(train_df))]
    test_set = train_df.iloc[int(train_validation_split * len(train_df)):]
    
    selected_features = random.sample(remaining_feature_set, max(3, int(len(remaining_feature_set) * initial_perc)))
    for feature in selected_features:
        remaining_feature_set.remove(feature)

    best_performer = -9999999999

    while 1:
        results = {}
        for feature in selected_features:
            if len(selected_features) < 2:
                break
            selected_features_test = selected_features[:]
            selected_features_test.remove(feature)
            clf.fit(train_set[selected_features_test], train_set[target_variable])
            results.update({feature : accuracy_score(clf.predict(test_set[selected_features_test]), test_set[target_variable])})
            
        for feature in remaining_feature_set:
            clf.fit(train_set[selected_features + [feature]], train_set[target_variable])
            results.update({feature : accuracy_score(clf.predict(test_set[selected_features + [feature]]), test_set[target_variable])})

        if (best_performer * (1 + minimum_improvement_perc)) < results[max(results, key=results.get)]:
            best_performer = results[max(results, key=results.get)]
            selected_feature = max(results, key=results.get)
            if selected_feature in selected_features:
                selected_features.remove(selected_feature)
                remaining_feature_set.append(selected_feature)
            else:
                selected_features.append(selected_feature)
                remaining_feature_set.remove(selected_feature)
            if len(list(remaining_feature_set)) == 0:
                return train_df[selected_features], selected_features
        else:
            return train_df[selected_features], selected_features

## Bayesian optimization-based feature selection
### Objective function
def objective_selection(trial):
    weighted_df = train_df.copy()
    for i, feature in enumerate(list(train_df.drop(target_variable, axis = 1))):
        if trial.suggest_int(feature, 0, 1) == 0:
            weighted_df = weighted_df.drop(feature, axis = 1)
    features_list = list(weighted_df.drop(target_variable, axis = 1))

    if len(features_list) == 0:
        return 9999999

    train_set = weighted_df.iloc[:int(train_validation_split * len(weighted_df))]
    test_set = weighted_df.iloc[int(train_validation_split * len(weighted_df)):]

    clf.fit(train_set[features_list], train_set[target_variable])
    return 1 - accuracy_score(clf.predict(test_set[features_list]), test_set[target_variable])

def bayesian_optimization_selection(clf, train_df, n_trials = 200):
    study = optuna.create_study()
    study.optimize(objective_selection, n_trials=n_trials)

    # need to normalize for feature mean value
    features_list = list(train_df.drop(target_variable, axis = 1))
    selected_features = []
    for feature in features_list:
        if study.best_trial.params[feature]:
            selected_features.append(feature)

    return train_df[selected_features], selected_features, study.best_trial.number

## Bayesian optimization-based feature weighting
### Objective function
def objective_weighting(trial):
    weighted_df = train_df.copy()
    features_list = list(train_df.drop(target_variable, axis = 1))
    for i, feature in enumerate(features_list):
        weighted_df[feature] *= trial.suggest_uniform(feature, 0, 1)

    train_set = weighted_df.iloc[:int(train_validation_split * len(weighted_df))]
    test_set = weighted_df.iloc[int(train_validation_split * len(weighted_df)):]

    clf.fit(train_set[features_list], train_set[target_variable])
    
    return 1 - accuracy_score(clf.predict(test_set[features_list]), test_set[target_variable])

def bayesian_optimization_weighting(clf, dataset, n_trials = 500):
    study = optuna.create_study()
    study.optimize(objective_weighting, n_trials=n_trials)

    return study.best_trial.params, study.best_trial.number

# Utils
def boolean_converter(features, dataset_features):
    to_append = []
    for feature in dataset_features:
        if feature in features:
            to_append.append(True)
        else:
            to_append.append(False)
    return to_append

def get_validation_result(clf, train_df, test_df, features, weights = False):
    train_set = train_df.copy()
    test_set = test_df.copy()
    if weights != False:
        for feature in features:
            if feature != target_variable and not np.isnan(weights[feature]):
                train_set[feature] *= weights[feature]
                test_set[feature] *= weights[feature]
    
    clf.fit(train_set[features], train_set[target_variable])

    pred = clf.predict(test_set[features])

    return accuracy_score(test_set[target_variable], pred), f1_score(test_set[target_variable], pred, average='macro')#, roc_auc_score(test_set[target_variable], pred, multi_class='ovr')


import openml
benchmark_suite = openml.study.get_suite('OpenML-CC18') # obtain the benchmark suite

# Results saver
#saver = open('results/openml_scores.csv', 'w')
#saver.write('cycle,task,method,instances,features,classes,accuracy,f1,optimization_time,prediction_time,no_features,best_trial\n')
#saver.close()
saver = open('results/openml_scores.csv', 'a')

# Check if process already occurred
psdf = pd.read_csv('results/openml_scores.csv')
already_processed = []
for _, row in psdf.iterrows():
    already_processed.append((row.cycle, row.task))

results = []
for cycle_id in range(n_iterations):
    for task_id in tqdm(benchmark_suite.tasks):  # iterate over all tasks
            
            if (cycle_id, task_id) in already_processed:
                continue

            # Write changes from previous cycle to disk
            saver.close()
            saver = open('results/openml_scores.csv', 'a')

            # Skipping this tasks since it is guaranteed that it will not run in under 3 hours
            if task_id == 3573:
                continue
            
            # Get OpenML task
            task = openml.tasks.get_task(task_id)
            features, targets = task.get_X_and_y()  

            # TODO Temporary skip
            #if len(features[0]) > 500 or len(features) > 50000:
            if len(features[0]) > 120 or len(features) >= 20000:
                continue

            # Get data into Pandas DataFrame format
            df = pd.DataFrame(features)
            df[target_variable] = targets
            df = df.dropna(axis = 0)

            # Shuffle data
            df = df.sample(frac=1)

            # Exclude dataset if removing NaNs substantially reduces data set
            if len(df) < 100:
                continue

            # Get list of data set features
            dataset_features = list(df.drop(target_variable, axis = 1)) 

            # Train&Validation-Test split
            split_point = int(len(df) * train_test_split)
            train_df = df.iloc[:split_point]
            
            # Normalizing features (mean 0, variance 1)
            df[dataset_features] = scale(df[dataset_features])
            train_df[dataset_features] = scale(train_df[dataset_features])

            # Get normalized test data
            test_df = df.iloc[split_point:]

            # Redefine classifier according to data set
            ## Guaranteeing min_neigbors < n_neighbors < max_neighbors
            n_neighbors_for_dataset = max(min_neighbors, min(max_neighbors, int(len(targets) * neighbor_samples_ratio)))
            clf = KNeighborsClassifier(n_neighbors = n_neighbors_for_dataset, n_jobs = -1)
            ## Limiting trials
            selection_trials = min(int(len(dataset_features) * 20), 400)
            weighting_trials = min(int(len(dataset_features) * 50), 1000)

            # KNN baseline
            try:
                ## Train phase
                t1 = time()
                t_optimize = time() - t1
                ## Test (no filter)
                performance_acc, performance_f1 = get_validation_result(clf, train_df, test_df, dataset_features)
                t_predict = time() - t1 - t_optimize
                saver.write(f'{cycle_id},{task_id},baseline,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},{performance_acc},{performance_f1},{t_optimize},{t_predict},{len(dataset_features)},0\n')
            except:
                saver.write(f'{cycle_id},{task_id},baseline,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},,,,,,0\n')


            # Intercorrelation filter
            try:
                ## Train phase
                t1 = time()
                _, features, weight_dict = intercorrelation_filter(train_df, threshold = 0.8)
                t_optimize = time() - t1
                ## Test filter
                performance_acc, performance_f1 = get_validation_result(clf, train_df, test_df, features)
                t_predict = time() - t1 - t_optimize
                saver.write(f'{cycle_id},{task_id},intercorrelation_filter,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},{performance_acc},{performance_f1},{t_optimize},{t_predict},{len(features)},0\n')
                ## Test weighting
                t1 = time()
                performance_acc, performance_f1 = get_validation_result(clf, train_df, test_df, features, weights = weight_dict)
                t_predict = time() - t1
                saver.write(f'{cycle_id},{task_id},correlation_weighting,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},{performance_acc},{performance_f1},{t_optimize},{t_predict},{len(features)},0\n')
            except:
                saver.write(f'{cycle_id},{task_id},intercorrelation_filter,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},,,,,,0\n')
                saver.write(f'{cycle_id},{task_id},correlation_weighting,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},,,,,,0\n')

            # Tree based filter
            try:
                ## Train phase
                t1 = time()
                _, features, weight_dict = tree_based_filter(train_df, feature_importance_threshold = 0.2)
                t_optimize = time() - t1
                ## Test filter
                performance_acc, performance_f1 = get_validation_result(clf, train_df, test_df, features)
                t_predict = time() - t1 - t_optimize
                saver.write(f'{cycle_id},{task_id},tree_based_filter,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},{performance_acc},{performance_f1},{t_optimize},{t_predict},{len(features)},0\n')
                ## Test weighting
                t1 = time()
                performance_acc, performance_f1 = get_validation_result(clf, train_df, test_df, features, weights = weight_dict)
                t_predict = time() - t1
                saver.write(f'{cycle_id},{task_id},tree_based_weighting,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},{performance_acc},{performance_f1},{t_optimize},{t_predict},{len(features)},0\n')
            except:
                saver.write(f'{cycle_id},{task_id},tree_based_filter,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},,,,,,0\n')
                saver.write(f'{cycle_id},{task_id},tree_based_weighting,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},,,,,,0\n')

            # l1 filter
            try:
                ## Train phase
                t1 = time()
                _, features = l1_filter(train_df, regularization_parameter = 0.01)
                t_optimize = time() - t1
                ## Test filter
                performance_acc, performance_f1 = get_validation_result(clf, train_df, test_df, features)
                t_predict = time() - t1 - t_optimize
                saver.write(f'{cycle_id},{task_id},l1filter,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},{performance_acc},{performance_f1},{t_optimize},{t_predict},{len(features)},0\n')
            except:
                saver.write(f'{cycle_id},{task_id},l1filter,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},,,,,,0\n')

            # ReliefF
            try:
                ## Train phase
                t1 = time()
                _, features, weight_dict = relieff(train_df, perc_features_to_keep = 0.75)
                t_optimize = time() - t1
                ## Test filter
                performance_acc, performance_f1 = get_validation_result(clf, train_df, test_df, features)
                t_predict = time() - t1 - t_optimize
                saver.write(f'{cycle_id},{task_id},relieff,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},{performance_acc},{performance_f1},{t_optimize},{t_predict},{len(features)},0\n')
                ## Test weighting
                t1 = time()
                performance_acc, performance_f1 = get_validation_result(clf, train_df, test_df, features, weights = weight_dict)
                t_predict = time() - t1
                saver.write(f'{cycle_id},{task_id},relieff_weighting,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},{performance_acc},{performance_f1},{t_optimize},{t_predict},{len(features)},0\n')
            except:
                saver.write(f'{cycle_id},{task_id},relieff,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},,,,,,0\n')
                saver.write(f'{cycle_id},{task_id},relieff_weighting,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},,,,,,0\n')

            # Forward selection
            try:
                ## Train phase
                t1 = time()
                _, features = forward_selection(clf, train_df, minimum_improvement_perc = 0.01)
                t_optimize = time() - t1
                ## Test filter
                performance_acc, performance_f1 = get_validation_result(clf, train_df, test_df, features)
                t_predict = time() - t1 - t_optimize
                saver.write(f'{cycle_id},{task_id},forward_selection,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},{performance_acc},{performance_f1},{t_optimize},{t_predict},{len(features)},0\n')
            except:
                saver.write(f'{cycle_id},{task_id},forward_selection,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},,,,,,0\n')

            # Backwards selection
            try:
                ## Train phase
                t1 = time()
                _, features = backwards_selection(clf, train_df, minimum_improvement_perc = 0.01)
                t_optimize = time() - t1
                ## Test filter
                performance_acc, performance_f1 = get_validation_result(clf, train_df, test_df, features)
                t_predict = time() - t1 - t_optimize
                saver.write(f'{cycle_id},{task_id},backwards_selection,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},{performance_acc},{performance_f1},{t_optimize},{t_predict},{len(features)},0\n')
            except:
                saver.write(f'{cycle_id},{task_id},backwards_selection,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},,,,,,0\n')

            # Step wise selection
            try:
                ## Train phase
                t1 = time()
                _, features = stepwise_selection(clf, train_df, minimum_improvement_perc = 0.01, initial_perc = 0.5)
                t_optimize = time() - t1
                ## Test filter
                performance_acc, performance_f1 = get_validation_result(clf, train_df, test_df, features)
                t_predict = time() - t1 - t_optimize
                saver.write(f'{cycle_id},{task_id},stepwise_selection,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},{performance_acc},{performance_f1},{t_optimize},{t_predict},{len(features)},0\n')
            except:
                saver.write(f'{cycle_id},{task_id},stepwise_selection,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},,,,,,0\n')

            # Bayesian optimization selection
            try:
                ## Train phase
                t1 = time()
                _, features, best_trial = bayesian_optimization_selection(clf, train_df, n_trials = selection_trials)
                t_optimize = time() - t1
                ## Test filter
                performance_acc, performance_f1 = get_validation_result(clf, train_df, test_df, features)
                t_predict = time() - t1 - t_optimize
                saver.write(f'{cycle_id},{task_id},bayesselection,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},{performance_acc},{performance_f1},{t_optimize},{t_predict},{len(features)},{best_trial}\n')
            except:
                saver.write(f'{cycle_id},{task_id},bayesselection,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},,,,,,0\n')

            # Bayesian optimization weighting
            try:
                ## Train phase
                t1 = time()
                weight_dict, best_trial = bayesian_optimization_weighting(clf, train_df, n_trials = weighting_trials)
                t_optimize = time() - t1
                ## Test weighting
                performance_acc, performance_f1 = get_validation_result(clf, train_df, test_df, dataset_features, weights = weight_dict)
                t_predict = time() - t1 - t_optimize
                saver.write(f'{cycle_id},{task_id},bayeswweighting,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},{performance_acc},{performance_f1},{t_optimize},{t_predict},{len(dataset_features)},{best_trial}\n')
            except:
                saver.write(f'{cycle_id},{task_id},bayeswweighting,{len(df)},{len(dataset_features)},{df[target_variable].nunique()},,,,,,0\n')

saver.close()
