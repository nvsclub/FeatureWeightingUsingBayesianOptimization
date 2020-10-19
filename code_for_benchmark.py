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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score

import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

from tqdm import tqdm

# Setup
task = 'Classification'

target_variable = 'target'
neighbor_samples_ratio = 0.05
train_validation_split = 0.75
train_test_split = 0.75
n_iterations = 10
offset_iterations = 0

# Defining the methods
## Bayesian optimization-based feature weighting
### Objective function
def objective_weighting(trial):
    weighted_df = dataset.copy()
    features_list = list(dataset.drop(target_variable, axis = 1))
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

def get_validation_result(clf, dataset, features, weights = False):
    if weights != False:
        for feature in list(dataset):
            if feature != target_variable and not np.isnan(weights[feature]):
                dataset[feature] *= weights[feature]

    train_set = dataset.iloc[:int(train_test_split * len(dataset))]
    test_set = dataset.iloc[int(train_test_split * len(dataset)):]

    clf.fit(train_set[features], train_set[target_variable])

    pred = clf.predict(test_set[features])

    return accuracy_score(test_set[target_variable], pred)


import openml
import signal

def timeout_handler(signum, frame):
    raise Exception('timeout')

benchmark_suite = openml.study.get_suite('OpenML-CC18') # obtain the benchmark suite

results = []
for cycle_id in range(5):
    for task_id in benchmark_suite.tasks:  # iterate over all tasks
        try:
            # Skipping this tasks since it is guaranteed that it will not run in under 3 hours
            if task_id == 3573:
                continue

            task = openml.tasks.get_task(task_id)  # download the OpenML task
            features, targets = task.get_X_and_y()  

            clf = KNeighborsClassifier(n_neighbors = min(50, int(len(targets) * neighbor_samples_ratio)), n_jobs = -1)
            clf2 = RandomForestClassifier(n_estimators = min(1000, int(len(targets))), n_jobs = -1)
            
            # Train&Validation-Test split
            split_point = int(len(targets) * train_test_split)
            features_train = features[:split_point]
            features_test = features[split_point:]
            targets_train = targets[:split_point]
            targets_test = targets[split_point:]

            if np.isnan(features_train).any() or np.isnan(features_test).any() or np.isnan(targets_train).any() or np.isnan(targets_test).any():
                continue

            split_dataset = pd.DataFrame(features_train)
            split_dataset[target_variable] = targets_train
            test_df = pd.DataFrame(features_test)
            test_df[target_variable] = targets_test
            dataset = split_dataset.append(test_df)

            dataset_features = list(dataset.drop(target_variable, axis = 1)) # Get list of data set features

            # Enabling running time out
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10800)

            # Normalizing features (mean 0, variance 1)
            split_dataset[dataset_features] = scale(split_dataset[dataset_features])
            dataset[dataset_features] = scale(dataset[dataset_features])
            
            # Remove possible NaN values
            split_dataset = split_dataset.dropna(axis = 1)
            dataset = dataset.dropna(axis = 1)


            # Get baselines
            ## RF simple
            t1 = time()
            performance_rf = get_validation_result(clf2, dataset, list(dataset.drop(target_variable, axis = 1)))
            t2_rf = time() - t1

            ## KNN simple
            t1 = time()
            performance_knn = get_validation_result(clf, dataset, list(dataset.drop(target_variable, axis = 1)))
            t2_knn = time() - t1

            # Bayesian optimization weighting
            t1 = time()
            weight_dict, best_trial = bayesian_optimization_weighting(clf, split_dataset, n_trials = 500)
            performance_bw = get_validation_result(clf, dataset, list(dataset.drop(target_variable, axis = 1)), weights = weight_dict)
            t2_bw = time() - t1

            # Disable time out
            signal.alarm(0)

            # Save results
            results.append([cycle_id, task_id, len(targets), len(dataset_features), performance_rf, performance_knn, performance_bw, t2_rf, t2_knn, t2_bw])
            print([cycle_id, task_id, len(targets), len(dataset_features), performance_rf, performance_knn, performance_bw, t2_rf, t2_knn, t2_bw])
            pd.DataFrame(results, columns=['cycle_id','task_id','len_data','len_features','perfRF','perfKNN','perfBW','timeRF','timeKNN','timeBW']).to_csv('results/openml_benchmarks.csv')
        
        except:
            print(task_id, 'Collapsed')