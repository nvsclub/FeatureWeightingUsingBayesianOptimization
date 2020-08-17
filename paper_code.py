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
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score

from ReliefF import ReliefF

import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

from tqdm import tqdm


# Setup
task = ['Regression', 'Classification']
task = task[1]

target_variable = 'target'
neighbor_samples_ratio = 0.05
train_validation_split = 0.75
train_test_split = 0.75
n_iterations = 10
offset_iterations = 0

# Loading datasets
dsets = {}

# For regression
if task == 'Regression':
    # Abalone (from UCI)
    abalone_data = pd.read_csv('data/abalone.csv')
    dsets.update( {'abalone' : abalone_data.sample(frac=1)} )

    # Bike sharing (from UCI)
    bike_data = pd.read_csv('data/bikesharing-day.csv').drop(['instant','dteday','casual','registered'], axis = 1)
    dsets.update( {'bike' : bike_data.sample(frac=1)} )

    # Boston housing (from pandas)
    boston_data = datasets.load_boston()
    df_boston = pd.DataFrame(boston_data.data,columns=boston_data.feature_names)
    df_boston['target'] = pd.Series(boston_data.target)
    dsets.update( {'boston' : df_boston.sample(frac=1)} )

    # Diabetes (from pandas)
    diabetes_data = datasets.load_diabetes()
    df_diabetes = pd.DataFrame(diabetes_data.data,columns=diabetes_data.feature_names)
    df_diabetes['target'] = pd.Series(diabetes_data.target)
    dsets.update( {'diabetes' : df_diabetes.sample(frac=1)} )

    # Forest fires (from UCI)
    fire_data = pd.read_csv('data/forestfires.csv')
    d = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    fire_data.month = fire_data.month.map(d)
    d = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5, 'sat':6, 'sun':7}
    fire_data.day = fire_data.day.map(d)
    fire_data.target = np.log(fire_data.target+1) # log normalization recomended by the author
    dsets.update( {'fire' : fire_data.sample(frac=1)} )
    
    # Machine (from UCI)
    machine_data = pd.read_csv('data/machine.csv')
    dsets.update( {'machine' : machine_data.sample(frac=1)} )

    # Student performance (from UCI)
    student_data = pd.read_csv('data/student-mat.csv', sep = ';')
    student_data.school = (student_data.school == 'GP') * 1
    student_data.sex = (student_data.sex == 'F') * 1
    student_data.address = (student_data.address == 'U') * 1
    student_data.famsize = (student_data.famsize == 'LE3') * 1
    student_data.Pstatus = (student_data.Pstatus == 'T') * 1
    student_data.Mjob = (student_data.Mjob == 'teacher') * 1
    student_data.Fjob = (student_data.Fjob == 'teacher') * 1
    student_data.reason = (student_data.reason == 'home') * 1
    student_data.schoolsup = (student_data.schoolsup == 'yes') * 1
    student_data.famsup = (student_data.famsup == 'yes') * 1
    student_data.paid = (student_data.paid == 'yes') * 1
    student_data.activities = (student_data.activities == 'yes') * 1
    student_data.nursery = (student_data.nursery == 'yes') * 1
    student_data.higher = (student_data.higher == 'yes') * 1
    student_data.internet = (student_data.reason == 'yes') * 1
    student_data.romantic = (student_data.romantic == 'yes') * 1
    student_data.guardian = (student_data.guardian == 'mother') * 1
    student_data = student_data.drop(['G1','G2'], axis = 1)
    dsets.update( {'student' : student_data.sample(frac=1)} )

elif task == 'Classification':
    # Australian (from UCI)
    australian_data = pd.read_csv('data/australian.csv')
    dsets.update( {'australian' : australian_data.sample(frac=1)} )


    # Glass (from UCI)
    glass_data = pd.read_csv('data/glass.csv').drop('a1', axis = 1)
    dsets.update( {'glass' : glass_data.sample(frac=1)} )

    # Breast cancer (from pandas)
    breast_cancer_data = datasets.load_breast_cancer()
    df_breast_cancer = pd.DataFrame(breast_cancer_data.data,columns=breast_cancer_data.feature_names)
    df_breast_cancer['target'] = pd.Series(breast_cancer_data.target)
    dsets.update( {'breast_cancer' : df_breast_cancer.sample(frac=1)} )

    # Ecoli (from UCI)
    ecoli_data = pd.read_csv('data/ecoli.csv').drop('a1', axis = 1)
    dsets.update( {'ecoli' : ecoli_data.sample(frac=1)} )

    # Ionosphere (from UCI)
    ionosphere_data = pd.read_csv('data/ionosphere.csv')
    dsets.update( {'ionosphere' : ionosphere_data.sample(frac=1)} )

    # Iris (from pandas)
    iris_data = datasets.load_iris()
    df_iris = pd.DataFrame(iris_data.data,columns=iris_data.feature_names)
    df_iris['target'] = pd.Series(iris_data.target)
    dsets.update( {'iris' : df_iris.sample(frac=1)} )

    # Sonar (from UCI)
    sonar_data = pd.read_csv('data/sonar.csv')
    dsets.update( {'sonar' : sonar_data.sample(frac=1)} )

    # Wine (from pandas)
    wine_data = datasets.load_wine()
    df_wine = pd.DataFrame(wine_data.data,columns=wine_data.feature_names)
    df_wine['target'] = pd.Series(wine_data.target)
    dsets.update( {'wine' : df_wine.sample(frac=1)} )

    # Car (from UCI) test for one hot encoded variables
    car_data = pd.read_csv('data/car.csv')
    car_features = car_data.drop('target', axis = 1)
    enc = OneHotEncoder()
    enc.fit(car_features)
    enc_car_data = pd.DataFrame(enc.transform(car_features).toarray(), columns = list(enc.get_feature_names()))
    enc_car_data['target'] = car_data.target
    dsets.update( {'car' : enc_car_data.sample(frac=1)} )

# Defining the methods
## Filter
### Eliminate low variance features
def variance_threshold_filter(dataset, threshold = 0.8):
    feature_set = dataset.drop(target_variable, axis = 1)
    sel = VarianceThreshold(threshold=(threshold * (1 - threshold)))
    sel.fit(feature_set)
    features_idx = sel.get_support()
    features = feature_set.columns[features_idx]
    return dataset[features], features

### Eliminate high intercorrelation features
def intercorrelation_filter(dataset, threshold = 0.8):
    feature_set = dataset.drop(target_variable, axis = 1)
    corr_matrix = abs(feature_set.corr())
    target_corr_matrix = abs(dataset.corr())
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
        weight_dict.update({feature : abs(dataset[feature].corr(dataset[target_variable]))})

    return dataset.drop('target', axis = 1)[selected_features], selected_features, weight_dict

## Embedded
### Random forest importance
def tree_based_filter(dataset, feature_importance_threshold = 0.2):
    if task in ['Classification', 'C']:
        clf = RandomForestClassifier()
    elif task in ['Regression', 'R']:
        clf = RandomForestRegressor()
    
    feature_set = dataset.drop(target_variable, axis = 1)
    features = list(feature_set)
    clf.fit(feature_set, dataset[target_variable])
    
    weight_dict = {}
    for feature, feature_importance in zip(features, clf.feature_importances_):
        weight_dict.update({feature : feature_importance})
    for feature, feature_importance in zip(features, clf.feature_importances_):
        if feature_importance < feature_importance_threshold:
            features.remove(feature)

    return dataset[features], features, weight_dict

### L1 regularization
def l1_filter(dataset, regularization_parameter = 0.01):
    feature_set = dataset.drop(target_variable, axis = 1)

    if task in ['Classification', 'C']:
        clf = LinearSVC(C=regularization_parameter, penalty="l1", dual=False).fit(feature_set, dataset[[target_variable]])
    elif task in ['Regression', 'R']:
        clf = Lasso().fit(feature_set, dataset[[target_variable]])

    model = SelectFromModel(clf, prefit=True)
    features_idx = model.get_support()
    features = feature_set.columns[features_idx]
    return dataset[features], features

### ReliefF
def relieff(dataset, perc_features_to_keep = 0.75):

    feature_set = dataset.drop(target_variable, axis = 1)
    feature_list = list(feature_set)

    train_set = dataset.iloc[:int(train_validation_split * len(dataset))]
    test_set = dataset.iloc[int(train_validation_split * len(dataset)):]

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

    return dataset[selected_features], selected_features, feature_weights

## Wrapper methods
### Forward selection
def forward_selection(clf, dataset, minimum_improvement_perc = 0.01):
    feature_set = dataset.drop(target_variable, axis = 1)
    train_set = dataset.iloc[:int(train_validation_split * len(dataset))]
    test_set = dataset.iloc[int(train_validation_split * len(dataset)):]
    selected_features = []
    
    best_performer = -9999999999
    if task in ['Regression', 'R']:
        minimum_improvement_perc = -minimum_improvement_perc

    while 1:
        results = {}
        for feature in feature_set:
            clf.fit(train_set[selected_features + [feature]], train_set[target_variable])
            if task in ['Classification', 'C']:
                results.update({feature : accuracy_score(clf.predict(test_set[selected_features + [feature]]), test_set[target_variable])})
            elif task in ['Regression', 'R']:
                results.update({feature : -mean_squared_error(clf.predict(test_set[selected_features + [feature]]), test_set[target_variable])})

        if (best_performer * (1 + minimum_improvement_perc)) < results[max(results, key=results.get)]:
            best_performer = results[max(results, key=results.get)]
            selected_features.append(max(results, key=results.get))
            feature_set.drop(selected_features[-1], axis = 1)
            if len(list(feature_set)) == 0:
                return dataset[selected_features], selected_features
        else:
            return dataset[selected_features], selected_features

### Backwards selection
def backwards_selection(clf, dataset, minimum_improvement_perc = 0.01):
    feature_set = dataset.drop(target_variable, axis = 1)
    train_set = dataset.iloc[:int(train_validation_split * len(dataset))]
    test_set = dataset.iloc[int(train_validation_split * len(dataset)):]
    selected_features = list(feature_set)

    best_performer = -9999999999
    if task in ['Regression', 'R']:
        minimum_improvement_perc = -minimum_improvement_perc

    while 1:
        results = {}
        for feature in selected_features:
            selected_features_test = selected_features[:]
            selected_features_test.remove(feature)
            clf.fit(train_set[selected_features_test], train_set[target_variable])
            if task in ['Classification', 'C']:
                results.update({feature : accuracy_score(clf.predict(test_set[selected_features_test]), test_set[target_variable])})
            elif task in ['Regression', 'R']:
                results.update({feature : -mean_squared_error(clf.predict(test_set[selected_features_test]), test_set[target_variable])})

        if (best_performer * (1 + minimum_improvement_perc)) < results[max(results, key=results.get)]:
            best_performer = results[max(results, key=results.get)]
            selected_features.remove(max(results, key=results.get))
            feature_set.drop(selected_features[-1], axis = 1)
            if len(list(feature_set)) == 0:
                return dataset[selected_features], selected_features
        else:
            return dataset[selected_features], selected_features

### Stepwise selection
def stepwise_selection(clf, dataset, minimum_improvement_perc = 0.01, initial_perc = 0.5):
    feature_set = dataset.drop(target_variable, axis = 1)
    train_set = dataset.iloc[:int(train_validation_split * len(dataset))]
    test_set = dataset.iloc[int(train_validation_split * len(dataset)):]
    
    selected_features = random.sample(list(feature_set), int(len(list(feature_set)) * initial_perc))
    for feature in selected_features:
        feature_set.drop(feature, axis = 1)

    best_performer = -9999999999
    if task in ['Regression', 'R']:
        minimum_improvement_perc = -minimum_improvement_perc

    while 1:
        results = {}
        for feature in selected_features:
            if len(selected_features) < 2:
                break
            selected_features_test = selected_features[:]
            selected_features_test.remove(feature)
            clf.fit(train_set[selected_features_test], train_set[target_variable])
            if task in ['Classification', 'C']:
                results.update({feature : accuracy_score(clf.predict(test_set[selected_features_test]), test_set[target_variable])})
            elif task in ['Regression', 'R']:
                results.update({feature : -mean_squared_error(clf.predict(test_set[selected_features_test]), test_set[target_variable])})

        for feature in feature_set:
            clf.fit(train_set[selected_features + [feature]], train_set[target_variable])
            if task in ['Classification', 'C']:
                results.update({feature : accuracy_score(clf.predict(test_set[selected_features + [feature]]), test_set[target_variable])})
            elif task in ['Regression', 'R']:
                results.update({feature : -mean_squared_error(clf.predict(test_set[selected_features + [feature]]), test_set[target_variable])})

        if (best_performer * (1 + minimum_improvement_perc)) < results[max(results, key=results.get)]:
            best_performer = results[max(results, key=results.get)]
            selected_feature = max(results, key=results.get)
            if selected_feature in selected_features:
                selected_features.remove(selected_feature)
            else:
                selected_features.append(selected_feature)
            feature_set.drop(selected_features[-1], axis = 1)
            if len(list(feature_set)) == 0:
                return dataset[selected_features], selected_features
        else:
            return dataset[selected_features], selected_features

## Bayesian optimization-based feature selection
### Objective function
def objective_selection(trial):
    weighted_df = dataset.copy()
    features_list = list(dataset.drop(target_variable, axis = 1))
    for i, feature in enumerate(features_list):
        if trial.suggest_int(feature, 0, 1) == 0:
            weighted_df = weighted_df.drop(feature, axis = 1)
    features_list = list(weighted_df.drop(target_variable, axis = 1))

    if len(features_list) == 0:
        return 99999999999

    train_set = weighted_df.iloc[:int(train_validation_split * len(weighted_df))]
    test_set = weighted_df.iloc[int(train_validation_split * len(weighted_df)):]

    clf.fit(train_set[features_list], train_set[target_variable])
    if task in ['Classification', 'C']:
        return 1 - accuracy_score(clf.predict(test_set[features_list]), test_set[target_variable])
    elif task in ['Regression', 'R']:
        return mean_squared_error(clf.predict(test_set[features_list]), test_set[target_variable])

def bayesian_optimization_selection(clf, dataset, n_trials = 500):
    study = optuna.create_study()
    study.optimize(objective_selection, n_trials=n_trials)

    # need to normalize for feature mean value
    features_list = list(dataset.drop(target_variable, axis = 1))
    selected_features = []
    for feature in features_list:
        if study.best_trial.params[feature]:
            selected_features.append(feature)

    return dataset[selected_features], selected_features, study.best_trial.number

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
    if task in ['Classification', 'C']:
        return 1 - accuracy_score(clf.predict(test_set[features_list]), test_set[target_variable])
    elif task in ['Regression', 'R']:
        return mean_squared_error(clf.predict(test_set[features_list]), test_set[target_variable])

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

    if task in ['Classification', 'C']:
        return [accuracy_score(test_set[target_variable], pred), f1_score(test_set[target_variable], pred, average='micro')]
    elif task in ['Regression', 'R']:
        return [mean_squared_error(test_set[target_variable], pred), r2_score(test_set[target_variable], pred)]

# Run main
for iteration in tqdm(range(offset_iterations, offset_iterations+n_iterations)):
    # Saver list for performance results
    results = []
    for dataset_index in dsets.keys():
        # Saver list for detailed results
        results_detailed = []

        dataset = dsets[dataset_index].sample(frac=1) # Randomize the data set order
        dataset = dataset.dropna(axis = 1)
        dataset_features = list(dataset.drop(target_variable, axis = 1)) # Get list of data set features

        # Initialize the classifiers
        if task == 'Classification':
            clf = KNeighborsClassifier(n_neighbors = int(len(dataset) * neighbor_samples_ratio), n_jobs = -1)
            clf2 = RandomForestClassifier(n_estimators = int(len(dataset)), n_jobs = -1)
        elif task == 'Regression':
            clf = KNeighborsRegressor(n_neighbors = int(len(dataset) * neighbor_samples_ratio), n_jobs = -1)
            clf2 = RandomForestRegressor(n_estimators = int(len(dataset)), n_jobs = -1)

        # Train&Validation-Test split
        split_dataset = dataset.iloc[:int(len(dataset) * train_test_split)].copy()
        # Normalizing features (mean 0, variance 1)
        split_dataset[dataset_features] = scale(split_dataset[dataset_features])
        dataset[dataset_features] = scale(dataset[dataset_features])
        # Remove possible NaN values
        split_dataset = split_dataset.dropna(axis = 1)
        dataset = dataset.dropna(axis = 1)

        # Get baselines
        ## KNN simple
        performance = get_validation_result(clf, dataset, list(dataset.drop(target_variable, axis = 1)))
        results.append([dataset_index, 'knn', 'all'] + performance + [0, 0])
        results_detailed.append([dataset_index, '001 knn', '000 all'] + performance + [0, 0] + [True for _ in dataset_features])
        ## RF simple
        performance = get_validation_result(clf2, dataset, list(dataset.drop(target_variable, axis = 1)))
        results.append([dataset_index, 'rf', 'all'] + performance + [0, 0])
        results_detailed.append([dataset_index, '000 rf', '000 all'] + performance + [0, 0] + [True for _ in dataset_features])

        # Variance
        ## Ignoring data set where errors occur
        try:
            timer0 = time()
            _, features = variance_threshold_filter(split_dataset, threshold = 0.8)
            timer0 = time() - timer0
            performance = get_validation_result(clf, dataset, features)
            results.append([dataset_index, '001 knn', '001 variance threshold'] + performance + [timer0, 0])
            results_detailed.append([dataset_index, '001 knn', '001 variance threshold'] + performance + [timer0, 0] + boolean_converter(features, dataset_features))
        except:
            results.append([dataset_index, '001 knn', '001 variance threshold'] + [-1, -1, -1, 0])
            results_detailed.append([dataset_index, '001 knn', '001 variance threshold'] + [-1, -1, -1, 0] + [False for _ in dataset_features])

        # Intercorrelation
        if task == 'Regression':
            timer0 = time()
            _, features, weight_dict = intercorrelation_filter(split_dataset, threshold = 0.8)
            timer0 = time() - timer0
            performance = get_validation_result(clf, dataset, features)
            results.append([dataset_index, '001 knn', '002 intercorrelation'] + performance + [timer0, 0])
            results_detailed.append([dataset_index, '001 knn', '002 intercorrelation'] + performance + [timer0, 0] + boolean_converter(features, dataset_features))
            performance = get_validation_result(clf, dataset, dataset_features, weights = weight_dict)
            results.append([dataset_index, '001 knn', '010 correlation weighting'] + performance + [timer0, 0])
            results_detailed.append([dataset_index, '001 knn', '010 correlation weighting'] + performance + [timer0, 0] + [weight_dict[feature] for feature in dataset_features])

        # Random forest feature importance
        timer0 = time()
        _, features, weight_dict = tree_based_filter(split_dataset, feature_importance_threshold = 0.2)
        timer0 = time() - timer0
        performance = get_validation_result(clf, dataset, features)
        results.append([dataset_index, '001 knn', '003 feature importance selection'] + performance + [timer0, 0])
        results_detailed.append([dataset_index, '001 knn', '003 feature importance selection'] + performance + [timer0, 0] + boolean_converter(features, dataset_features))
        performance = get_validation_result(clf, dataset, list(dataset.drop(target_variable, axis = 1)), weights = weight_dict)
        results.append([dataset_index, '001 knn', '011 feature importance weighting'] + performance + [timer0, 0])
        results_detailed.append([dataset_index, '001 knn', '011 feature importance weighting'] + performance + [timer0, 0] + [weight_dict[feature] for feature in dataset_features])
        
        # L1 regularization test
        try:
            timer0 = time()
            _, features = l1_filter(split_dataset, regularization_parameter = 0.01)
            timer0 = time() - timer0
            performance = get_validation_result(clf, dataset, features)
            results.append([dataset_index, '001 knn', '004 l1'] + performance + [timer0, 0])
            results_detailed.append([dataset_index, '001 knn', '004 l1'] + performance + [timer0, 0] + boolean_converter(features, dataset_features))
        except:
            results.append([dataset_index, '001 knn', '004 l1'] + [-1, -1, 0, 0])
            results_detailed.append([dataset_index, '001 knn', '004 l1'] + [-1, -1, 0, 0] + [False for _ in dataset_features])
        
        # ReliefF
        timer0 = time()
        _, features, weight_dict = relieff(split_dataset, perc_features_to_keep = 0.75)
        timer0 = time() - timer0
        performance = get_validation_result(clf, dataset, features)
        results.append([dataset_index, '001 knn', '005 relieff selection'] + performance + [timer0, 0])
        results_detailed.append([dataset_index, '001 knn', '005 relieff selection'] + performance + [timer0, 0] + boolean_converter(features, dataset_features))
        performance = get_validation_result(clf, dataset, list(dataset.drop(target_variable, axis = 1)), weights = weight_dict)
        results.append([dataset_index, '001 knn', '012 relieff weighting'] + performance + [timer0, 0])
        results_detailed.append([dataset_index, '001 knn', '012 relieff weighting'] + performance + [timer0, 0] + [weight_dict[feature] for feature in dataset_features])

        # Forward selection
        timer0 = time()
        _, features = forward_selection(clf, split_dataset, minimum_improvement_perc = 0.01)
        timer0 = time() - timer0
        performance = get_validation_result(clf, dataset, features)
        results.append([dataset_index, '001 knn', '006 forward selection'] + performance + [timer0, 0])
        results_detailed.append([dataset_index, '001 knn', '006 forward selection'] + performance + [timer0, 0] + boolean_converter(features, dataset_features))
        
        # Backwards selection
        timer0 = time()
        _, features = backwards_selection(clf, split_dataset, minimum_improvement_perc = 0.01)
        timer0 = time() - timer0
        performance = get_validation_result(clf, dataset, features)
        results.append([dataset_index, '001 knn', '007 backward selection'] + performance + [timer0, 0])
        results_detailed.append([dataset_index, '001 knn', '007 backward selection'] + performance + [timer0, 0] + boolean_converter(features, dataset_features))
        
        # Stepwise selection
        timer0 = time()
        _, features = stepwise_selection(clf, split_dataset, minimum_improvement_perc = 0.01, initial_perc = 0.5)
        timer0 = time() - timer0
        performance = get_validation_result(clf, dataset, features)
        results.append([dataset_index, '001 knn', '008 stepwise selection'] + performance + [timer0, 0])
        results_detailed.append([dataset_index, '001 knn', '008 stepwise selection'] + performance + [timer0, 0] + boolean_converter(features, dataset_features))
        
        # Bayesian optimization weighting
        timer0 = time()
        _, features, best_trial = bayesian_optimization_selection(clf, split_dataset, n_trials = 200)
        timer0 = time() - timer0
        performance = get_validation_result(clf, dataset, features)
        results.append([dataset_index, '001 knn', '009 bayesian selection'] + performance + [timer0, best_trial])
        results_detailed.append([dataset_index, '001 knn', '009 bayesian selection'] + performance + [timer0, best_trial] + boolean_converter(features, dataset_features))
        
        # Bayesian optimization weighting
        timer0 = time()
        weight_dict, best_trial = bayesian_optimization_weighting(clf, split_dataset, n_trials = 500)
        timer0 = time() - timer0
        performance = get_validation_result(clf, dataset, list(dataset.drop(target_variable, axis = 1)), weights = weight_dict)
        results.append([dataset_index, '001 knn', '013 bayesian weighting'] + performance + [timer0, best_trial])
        results_detailed.append([dataset_index, '001 knn', '013 bayesian weighting'] + performance + [timer0, best_trial] + [weight_dict[feature] for feature in dataset_features])

        # Register detailed results
        results_register = pd.DataFrame(results_detailed, columns = ['dataset', 'base', 'method'] + ['PM1', 'PM2', 'Time', 'TriesToOpt'] + dataset_features)
        results_register.to_csv('results/detailed/' + dataset_index + '_' + str(iteration) + '.csv')

    # Register performance results
    results_register = pd.DataFrame(results, columns = ['dataset', 'base', 'method'] + ['PM1', 'PM2', 'Time', 'TriesToOpt'])
    results_register.to_csv('results/aggregated/res_' + task + '_' + str(iteration) + '.csv')

