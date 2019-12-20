from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import datetime
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from utility_functions import *
from scipy import stats

#from sklearn import cross_validation, linear_model, svm
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.metrics import confusion_matrix, roc_curve, auc
#from sklearn.ensemble import ExtraTreesClassifier,BaggingClassifier,RandomForestClassifier,AdaBoostClassifier
#from sklearn.ensemble.forest import RandomForestRegressor
#from sklearn import cross_validation
#import operator
#import affinity
#import multiprocessing as mp
#affinity.set_process_affinity_mask(0,2**mp.cpu_count()-1)
#os.system("taskset -p 0xff %d" % os.getpid())

def generate_multi_output_regression_model(data_array
                               , param_space
                               , Xcols
                               , Ycols
                               , learn_ratio
                               , debug=False
                               , n_estimators=10
                               , max_features=0.5
                               , customRegressor=RandomForestRegressor
                               , print_importances=False):
    """
    Fit a Random Forest model (for now it is Random Forest but in the future we will host more models here (e.g. GPs and lattices).
    This method fits a single multi-output model for all objectives.
    :param data_array: the data to use for training.
    :param Xcols: the names of the input features used for training.
    :param Ycols: the names of the output labels used for training.
    :param learn_ratio: percentage of the input vectors used for training. A part of it maybe left over for cross validation.
    :param debug: is debugging mode enabled?
    :param n_estimators: number of trees.
    :param max_features: this is a parameter of the Random Forest. It decides how many feature to randomize.
    :param customRegressor: regression model to be used
    :return: 3 variables: the classifier, X_test , Y_test.
    """
    start_time = datetime.datetime.now()

    if param_space.get_input_normalization_flag() is True:
        compute_mean_and_std(data_array, param_space)
    preprocessed_data_array = preprocess_data_array(data_array, param_space, Xcols)
    X = [preprocessed_data_array[param] for param in preprocessed_data_array]
    X = list(map(list, list(zip(*X))))
    Y = [data_array[Ycol] for Ycol in Ycols]
    Y = list(map(list, list(zip(*Y))))

    learn_size = int(len(X) * learn_ratio)
    X_train = X[0:learn_size]
    X_test  = X[learn_size:]
    y_train = Y[0:learn_size]
    Y_test = Y[learn_size:]

    if len (X_test) == 0:
        X_test = X[:]
    if len (Y_test) == 0:
        Y_test = Y[:]

    regressor = customRegressor(n_estimators=n_estimators, bootstrap=False, min_samples_split=5, max_features = max_features, n_jobs=1) # Parallel: n_jobs=-1 will use all available processors
    regressor.fit(X_train, y_train)

    if print_importances:
        parameter_importances = compute_parameter_importance(regressor, Xcols, param_space)
        print("Regression model on " + str(Ycols) + ". Features names: " + str(Xcols) + ", feature importances: " + str(parameter_importances))
    sys.stdout.write_to_logfile(("End of training - Time %10.2f sec\n" % ((datetime.datetime.now() - start_time).total_seconds())))

    return regressor, X_test , Y_test

def generate_mono_output_regression_models(data_array
                               , param_space
                               , Xcols
                               , Ycols
                               , learn_ratio
                               , debug=False
                               , n_estimators=10
                               , max_features=0.5
                               , customRegressor=RandomForestRegressor
                               , number_of_cpus=0
                               , print_importances=False):
    """
    Fit a Random Forest model (for now it is Random Forest but in the future we will host more models here (e.g. GPs and lattices).
    This method fits one mono output model for each objective.
    :param data_array: the data to use for training.
    :param Xcols: the names of the input features used for training.
    :param Ycols: the names of the output labels used for training.
    :param learn_ratio: percentage of the input vectors used for training. A part of it maybe left over for cross validation.
    :param debug: is debugging mode enabled?
    :param n_estimators: number of trees.
    :param max_features: this is a parameter of the Random Forest. It decides how many feature to randomize.
    :param customRegressor:
    :param number_of_cpus:
    :return: 3 variables: the classifier, X_test , Y_test.
    """
    start_time = datetime.datetime.now()

    if param_space.get_input_normalization_flag() is True:
        compute_mean_and_std(data_array, param_space)
    preprocessed_data_array = preprocess_data_array(data_array, param_space, Xcols)
    regressor_baggedtrees = {}
    X = [preprocessed_data_array[param] for param in preprocessed_data_array]
    X = list(map(list, list(zip(*X))))
    learn_size = int(len(X) * learn_ratio)
    X_train = X[0:learn_size]
    X_test  = X[learn_size:]
    Y_test  = {}

    if len (X_test) == 0:
        X_test = X[:]

    for i, Ycol in enumerate(Ycols):
        Y = data_array[Ycol]
        y_train      = Y[0:learn_size]
        Y_test[Ycol] = Y[learn_size:]
        if len (Y_test[Ycol]) == 0:
            Y_test[Ycol] = Y[:]

        if debug:
            print("Metric:%s, prepare training: len(X)=%s, len(X_train)=%s (learn_size=%s), len(X_test)=%s" % (
                                                                Ycol, len(X), len(X_train), learn_size, len(X_test)))
            print(("Prepare training packages len(X) = %s" % len(X)))
            if i == 0:
                print("X_train")
                print(X_train)
            print("Y_train")
            print(y_train)
            # X_train, X_test, y_train, y_test_accuracy = cross_validation.train_test_split(X, Yall, test_size = 0.33, random_state = 0)
            print("Run accuracy prediction training...")


        #print(("Affinity mask: %d"%affinity.get_process_affinity_mask(0)))
        #classifier_baggedtrees[Ycol] = customClassifier(n_estimators=n_estimators, max_features = max_features) # Sequential
        regressor_baggedtrees[Ycol] = customRegressor(n_estimators=n_estimators, bootstrap=False, min_samples_split=5, max_features = max_features, n_jobs=1) # Parallel: n_jobs=-1 will use all available processors
        regressor_baggedtrees[Ycol].fit(X_train, y_train)
        if print_importances:
            parameter_importances = compute_parameter_importance(regressor_baggedtrees[Ycol], Xcols, param_space)
            print("Regression model on " + str(Ycol) + ". Features names: " + str(Xcols) + ", feature importances: " + str(parameter_importances))

    sys.stdout.write_to_logfile(("End of training - Time %10.2f sec\n" % ((datetime.datetime.now() - start_time).total_seconds())))
    return regressor_baggedtrees, X_test , Y_test


def generate_classification_model(application_name
                                  , param_space
                                  , data_array
                                  , Xcols
                                  , Ycols
                                  , learn_ratio
                                  , debug=False
                                  , n_estimators=15
                                  , max_features=0.5
                                  , customClassifier=ExtraTreesRegressor
                                  , number_of_cpus=0
                                  , data_array_exhaustive=None
                                  , enable_feasible_predictor_grid_search_on_recall_and_precision=False
                                  , feasible_predictor_grid_search_validation_file=""
                                  , print_importances=False):
    """
    Fit a Random Forest model (for now it is Random Forest but in the future we will host more models here (e.g. GPs and lattices).
    :param application_name: the name of the application given by the json file.
    :param param_space:
    :param data_array: the data to use for training.
    :param Xcols: the names of the input features used for training.
    :param Ycols: the names of the output labels used for training.
    :param learn_ratio: percentage of the input vectors used for training. A part of it maybe left over for cross validation.
    :param debug: is debugging mode enabled?
    :param n_estimators: number of trees.
    :param max_features: this is a parameter of the Random Forest. It decides how many feature to randomize.
    :param customClassifier:
    :param number_of_cpus:
    :param enable_feasible_predictor_grid_search_on_recall_and_precision: does grid search on recall and precision to study the quality of the classifier.
    :param feasible_predictor_grid_search_validation_file: provides the dataset file name of the grid search cross-validation dataset.
    :return: 3 variables: the classifier, X_test , Y_test.
    """
    start_time = datetime.datetime.now()

    if param_space.get_input_normalization_flag() is True:
        compute_mean_and_std(data_array, param_space)
    preprocessed_data_array = preprocess_data_array(data_array, param_space, Xcols)

    classifier_baggedtrees = {}
    X = [preprocessed_data_array[param] for param in preprocessed_data_array]
    X = list(map(list, list(zip(*X))))
    learn_size = int(len(X) * learn_ratio)
    X_train = X[0:learn_size]
    X_test = X[learn_size:]
    Y_test = {}

    if len(X_test) == 0:
        X_test = X[:]

    for i, Ycol in enumerate(Ycols):
        Y = data_array[Ycol]
        y_train = Y[0:learn_size]
        Y_test[Ycol] = Y[learn_size:]
        if len(Y_test[Ycol]) == 0:
            Y_test[Ycol] = Y[:]

        if debug:
            print("Metric:%s, prepare training: len(X)=%s, len(X_train)=%s (learn_size=%s), len(X_test)=%s" % (
                Ycol, len(X), len(X_train), learn_size, len(X_test)))
            print(("Prepare training packages len(X) = %s" % len(X)))
            if i == 0:
                print("X_train")
                print(X_train)
            print("Y_train")
            print(y_train)
            # X_train, X_test, y_train, y_test_accuracy = cross_validation.train_test_split(X, Yall, test_size = 0.33, random_state = 0)
            print("Run accuracy prediction training...")

        # print(("Affinity mask: %d"%affinity.get_process_affinity_mask(0)))
        # classifier_baggedtrees[Ycol] = customClassifier(n_estimators=n_estimators, max_features = max_features) # Sequential
        class_weight = {True: 0.9, False: 0.1} # Default is class_weight=None
        classifier_baggedtrees[Ycol] = RandomForestClassifier(class_weight=class_weight, n_estimators=10, max_features=0.75) # n_estimators=10 is the default
        classifier_baggedtrees[Ycol].fit(X_train, y_train)

        if data_array_exhaustive != None:
            preprocessed_x_exhaustive = preprocess_data_array(data_array_exhaustive, param_space, Xcols)
            X_exhaustive = [preprocessed_x_exhaustive[param] for param in preprocessed_x_exhaustive]
            X_exhaustive = list(map(list, list(zip(*X_exhaustive))))
            for i, Ycol in enumerate(Ycols):
                y_exhaustive = data_array_exhaustive[Ycol]
                print("Score of the feasibility classifier: " + str(classifier_baggedtrees[Ycol].score(X_exhaustive, y_exhaustive)))

        if print_importances:
            parameter_importances = compute_parameter_importance(classifier_baggedtrees[Ycol], Xcols, param_space)
            print("Classification model. Features names: " + str(Xcols) + ", feature importances: " + str(parameter_importances))
        #customClassifier = RandomForestClassifier
        #classifier_baggedtrees[Ycol] = customClassifier(n_estimators=n_estimators, max_features=max_features,
        #                                                n_jobs=-1)  # Parallel: n_jobs=-1 will use all available processors
        #classifier_baggedtrees[Ycol].fit(X_train, y_train)

        if enable_feasible_predictor_grid_search_on_recall_and_precision:
            #dataset = "/Users/lnardi/Dropbox/sw/hypermapper/spatial_data/2018-04-16_21-43-25_reference_heuristic/BlackScholes/BlackScholes_heuristic/BlackScholes_heuristic_dse/BlackScholes_trial_0.csv"
            #dataset = "/home/lnardi/spatial-lang/results/apps_classification_test_set/" + application_name + ".csv"
            dataset = feasible_predictor_grid_search_validation_file
            compute_recall_and_precision_on_RF_hyperparameters(dataset, param_space, X_train, y_train)

    sys.stdout.write_to_logfile(("End of training - Time %10.2f sec\n" % ((datetime.datetime.now() - start_time).total_seconds())))
    return classifier_baggedtrees, X_test, Y_test

def compute_recall_and_precision_on_RF_hyperparameters(dataset, param_space, X_train, y_train):
    """
    Compute recall and precision for the binary random forests classifier using cross validation.
    Reference: https://en.wikipedia.org/wiki/Precision_and_recall
    This function should be used only for debugging and development purposes.
    For debugging to see if a new application performs well on the set of hyperparameters of the random forests binary classifier.
    For development to set these hyperparameters.
    The objective in HyperMapper is to maximize the recall because we don't want to loose the opportunity of finding good samples.
    At the same time we should keep under control the precision for efficiency (we don't want too many false positives
    because in this case the effect would be like not having the classifier at all).
    Notice that using the accuracy instead of the recall would be the wrong thing to do because we care more about the
    fact that we want to classify precisely the true class (we care less of the false class).
    :param dataset: the test data to use to check the recall and precision.
    :param param_space:
    :param X_train: the features data used to train the Random Forests.
    :param y_train: the labels data used to train the Random Forests.
    :return:
    """
    start_time = datetime.datetime.now()
    learn_ratio = 0.75
    Xcols = param_space.get_input_parameters()
    Ycol = param_space.get_feasible_parameter()[0]
    print("#######################################################################")
    print("####### Start of the cross-validation for the RF classifier ###########")
    print(("Loading data from %s ..." % dataset))
    data_array, fast_addressing_of_data_array = param_space.load_data_file(dataset)
    count = 0
    for i in data_array[Ycol]:
        if i == True:
            count += 1
    print("\nCount of feasible in the dataset file %s = %d" % (dataset, count))
    # Set the parameters by cross-validation
    tuned_parameters = [{
         'n_estimators': [10, 100, 1000], # 'n_estimators': [10, 20, 30, 50, 100, 1000],
         'max_features': ["auto", 0.5, 0.75], #'max_features': ["auto", 0.25, 0.5, 0.75],
         #'criterion': ["gini", "entropy"],
         'max_depth': [None, 4, 8], # 'max_depth': [2, 4, 6, 8, 10, 20],
         'class_weight': [{True: 0.50, False: 0.50}, {True: 0.75, False: 0.25}, {True: 0.9, False: 0.1}]
         # 'class_weight': [{True: 0.25, False: 0.75}, {True: 0.50, False: 0.50}, {True: 0.75, False: 0.25}, {True: 0.8, False: 0.2}, {True: 0.85, False: 0.15}, {True: 0.9, False: 0.1}, {True: 0.95, False: 0.05}]
    }]
    X_test = [data_array[Xcol] for Xcol in Xcols]
    X_test = list(map(list, list(zip(*X_test))))
    Y_test = data_array[Ycol]
    scores = ['recall', 'precision'] #scores = ['recall']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)

        clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, n_jobs=-1, scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found:")
        print(clf.best_params_)

        print("Grid scores:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        print("Detailed classification report:")
        print("The model is trained on the HyperMapper samples, total samples: %d" %len(X_train))
        print("The scores are computed on the test set.")
        y_true, y_pred = Y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))

    end_time = datetime.datetime.now()
    print(("\nTotal time to compute the cross-validation for the Random Forests binary classifier is: " + str((end_time - start_time).total_seconds()) + " seconds"))
    print("####### End of the cross-validation for the RF classifier ###########")
    print("#######################################################################")

def compute_parameter_importance(model, input_params, param_space):
    parameter_importances = [0]*len(input_params)
    categorical_parameters = param_space.get_input_categorical_parameters_objects(input_params)
    feature_idx = 0
    for idx, param in enumerate(input_params):
        if param in categorical_parameters:
            size = categorical_parameters[param].get_size()
            parameter_importances[idx] = sum(model.feature_importances_[feature_idx:feature_idx+size])
            feature_idx = feature_idx + size
        else:
            parameter_importances[idx] = model.feature_importances_[feature_idx]
            feature_idx += 1
    return parameter_importances


def parallel_model_prediction(model, bufferx, param_space, debug=False, number_of_cpus=0):
    """
    This function explicitly parallelize the prediction of the Random Forest model.
    The parallelization is done by hand however not exploiting the fact that the RF model has an option n_jobs that automatically parallize the fit and prediction.
    That option seems not to work though.
    :param model: model (or dictionary of models) to use for prediction.
    :param bufferx: data to perform prediction on.
    :param optimization_metrics: the names of the objectives being optimized.
    :param debug: turn debug mode on/off.
    :param number_of_cpus: number of cpus to use in parallel.
    :return:
    """
    if type(model) is dict:
        return domain_decomposition_and_parallel_computation(debug, mono_output_model_prediction, concatenate_function_prediction, bufferx, number_of_cpus, model, param_space)
    else:
        return domain_decomposition_and_parallel_computation(debug, multi_output_model_prediction, concatenate_function_prediction, bufferx, number_of_cpus, model, param_space)
    #return domain_decomposition_and_parallel_computation(debug, worker, concatenate_function_prediction, bufferx, classifier)

def multi_output_model_prediction(bufferx, model, param_space):
    """
    :param bufferx: list containing points to predict.
    :param model: model to use for prediction.
    :param optimization_metrics: list containing the metrics being optimized.
    :return: dictionary containing predictions for each objective.
    """
    optimization_metrics = param_space.get_optimization_parameters()
    input_params = param_space.get_input_parameters()
    normalized_bufferx = preprocess_data_buffer(bufferx, param_space)
    Cresult = {}
    predictions = model.predict(normalized_bufferx)
    for idx, parameter in enumerate(optimization_metrics):
        Cresult[parameter] = predictions[:,idx]
    return Cresult

def mono_output_model_prediction(bufferx, model, param_space):
    """
    :param bufferx: list containing points to predict.
    :param model: model to use for prediction.
    :return: dictionary containing predictions for each objective.
    """
    normalized_bufferx = preprocess_data_buffer(bufferx, param_space)
    Cresult = {}
    for parameter in model:
        Cresult[parameter] = model[parameter].predict(normalized_bufferx)
    return Cresult

def model_prediction(bufferx, model, param_space):
    """
    Compute the predictions of a model over a data array.
    :param bufferx: data array with points to be predicted.
    :param model: model to use to perform prediction.
    :return: array with model predictions.
    """
    normalized_bufferx = preprocess_data_buffer(bufferx, param_space)
    Cresult = {}
    for parameter in model:
        Cresult[parameter] = model[parameter].predict(normalized_bufferx)
    return Cresult

def model_probabilities(bufferx, model, param_space):
    """
    Compute the predictions of a model over a data array.
    :param bufferx: data array with points to be predicted.
    :param model: model to use to perform prediction.
    :return: array with model predictions.
    """
    normalized_bufferx = preprocess_data_buffer(bufferx, param_space)
    Cresult = {}
    for parameter in model:
        Cresult[parameter] = model[parameter].predict_proba(normalized_bufferx)
    return Cresult

def tree_predictions(bufferx, model, param_space):
    normalized_bufferx = preprocess_data_buffer(bufferx, param_space)
    optimization_metrics = param_space.get_optimization_parameters()
    tree_predictions = {}
    for metric in optimization_metrics:
        tree_predictions[metric] = np.array([tree.predict(normalized_bufferx) for tree in model[metric]])
    return tree_predictions

def get_leaves_per_sample(bufferx, model, param_space):
    normalized_bufferx = preprocess_data_buffer(bufferx, param_space)
    optimization_metrics = param_space.get_optimization_parameters()
    leaf_per_sample = {}
    for metric in optimization_metrics:
        leaf_per_sample[metric] = np.array([tree.apply(normalized_bufferx) for tree in model[metric]])
    return leaf_per_sample

def concatenate_function_prediction(results_parallel) :
    """
    Concatenate the results of parallel predictions into a single data dictionary.
    :param bufferx: data array with points to be predicted.
    :return: dictionary containing concatenated results.
    """
    concatenate_result = {}
    for key in list(results_parallel[0].keys()) :
        concatenate_result[key] = np.concatenate([results_parallel[chunk][key] for chunk in range(len(results_parallel))])
    return concatenate_result

def preprocess_data_buffer(bufferx, param_space):
    """
    Preprocess an input buffer before feeding into a regression/classification model.
    The preprocessing standardize non-categorical inputs (if the flag is set).
    It also transforms categorical variables using one-hot encoding.
    :param bufferx: data array containing the input configurations to preprocess.
    :param param_space: parameter space object for the current application.
    :return: preprocessed data buffer.
    """
    input_params = param_space.get_input_parameters()
    data_array = data_tuples_to_dictionary(bufferx, input_params)
    preprocessed_data_array = preprocess_data_array(data_array, param_space, input_params)
    preprocessed_buffer = data_dictionary_to_tuple(preprocessed_data_array, list(preprocessed_data_array.keys()))
    return preprocessed_buffer

def preprocess_data_array(data_array, param_space, input_params):
    """
    Preprocess a data_array before feeding into a regression/classification model.
    The preprocessing standardize non-categorical inputs (if the flag is set).
    It also transforms categorical variables using one-hot encoding.
    :param data_array: dictionary containing the input configurations to preprocess.
    :param param_space: parameter space object for the current application.
    :param input_params: list with the names of the input parameters to preprocess in the data array.
    :return: preprocessed data array. The returned data array will contain only the keys in input_params.
    """
    non_categorical_parameters = param_space.get_input_non_categorical_parameters(input_params)
    categorical_parameters = param_space.get_input_categorical_parameters_objects(input_params)
    preprocessed_data_array = {}
    for param in non_categorical_parameters:
        if param_space.get_input_normalization_flag() is True:
            X = np.array(data_array[param], dtype=np.float64)
            mean = param_space.get_parameter_mean(param)
            std = param_space.get_parameter_std(param)
            X = (X - mean)/std
            preprocessed_data_array[param] = X
        else:
            preprocessed_data_array[param] = data_array[param]
    for param in categorical_parameters:
        # Categorical variables are encoded as their index, generate a list of "index labels"
        categories = np.arange(categorical_parameters[param].get_size())
        encoder = OneHotEncoder(categories="auto", sparse=False)
        encoder.fit(categories.reshape(-1,1))
        x = np.array(data_array[param]).reshape(-1,1)
        encoded_x = encoder.transform(x)
        for i in range(encoded_x.shape[1]):
            new_key = param + "_" + str(categories[i])
            preprocessed_data_array[new_key] = list(encoded_x[:,i])
    return preprocessed_data_array

def compute_mean_and_std(data_array, param_space):
    input_params = param_space.get_input_parameters()
    for param in input_params:
        X = np.array(data_array[param], dtype=np.float64)
        mean = np.mean(X)
        std = np.std(X)
        param_space.set_parameter_mean(param, mean)
        param_space.set_parameter_std(param, std)

def get_mean_per_leaf(samples, leaf_per_sample):
    """
    Compute the mean value for each leaf in the forest.
    :param samples: list with the value of each sample used to build the forest.
    :param leaf_per_sample: matrix with dimensions number_of_trees * number_of_samples. Stores the leaf each sample fell into for each tree.
    :return: list of number_of_trees dictionaries. Each dictionary contains the means for each leaf in a tree.
    """
    number_of_trees, number_of_samples = leaf_per_sample.shape
    tree_means_per_leaf = []
    for tree_idx in range(number_of_trees):
        leaf_means = defaultdict(int)
        leaf_sample_count = defaultdict(int)
        for sample_idx in range(number_of_samples):
            leaf = leaf_per_sample[tree_idx, sample_idx]
            leaf_sample_count[leaf] += 1
            leaf_means[leaf] += samples[sample_idx]
        for leaf in leaf_sample_count.keys():
            leaf_means[leaf] = leaf_means[leaf]/leaf_sample_count[leaf]
        tree_means_per_leaf.append(leaf_means)
    return tree_means_per_leaf

def get_var_per_leaf(samples, leaf_per_sample):
    """
    Compute the variance for each leaf in the forest.
    :param samples: list with the value of each sample used to build the forest.
    :param leaf_per_sample: matrix with dimensions number_of_trees * number_of_samples. Stores the leaf each sample fell into for each tree.
    :return: list of number_of_trees dictionaries. Each dictionary contains the variance for each leaf in a tree.
    """
    number_of_trees, number_of_samples = leaf_per_sample.shape
    tree_vars_per_leaf = []
    for tree_idx in range(number_of_trees):
        samples_per_leaf = defaultdict(list)
        for sample_idx in range(number_of_samples):
            leaf = leaf_per_sample[tree_idx, sample_idx]
            samples_per_leaf[leaf].append(samples[sample_idx])

        leaf_vars = {}
        for leaf in samples_per_leaf.keys():
            if len(samples_per_leaf[leaf]) > 1:
                leaf_vars[leaf] = np.var(samples_per_leaf[leaf], ddof=1)
            else:
                leaf_vars[leaf] = 0
            # leaf_vars[leaf] = max(leaf_vars[leaf], 0.01) # This makes HyperMapper exploit too much. We will revisit this.
        tree_vars_per_leaf.append(leaf_vars)

    return tree_vars_per_leaf

def compute_rf_prediction(leaf_per_sample, leaf_means):
    """
    Compute the forest prediction for a list of samples based on the mean of the leaves in each tree.
    :param leaf_per_sample: matrix with dimensions number_of_trees * number_of_samples. Stores the leaf each sample fell into for each tree.
    :param leaf_means: list of number_of_trees dictionaries. Each dictionary contains the mean for each leaf in a tree.
    :return: list containing the mean of each sample.
    """
    number_of_trees, number_of_samples = leaf_per_sample.shape
    sample_means = np.zeros(number_of_samples)
    for tree_idx in range(number_of_trees):
        for sample_idx in range(number_of_samples):
            sample_leaf = leaf_per_sample[tree_idx, sample_idx]
            sample_means[sample_idx] += leaf_means[tree_idx][sample_leaf]/number_of_trees
    return sample_means

def compute_rf_prediction_variance(leaf_per_sample, sample_means, leaf_means, leaf_vars):
    """
    Compute the forest prediction variance for a list of samples based on the mean and variances of the leaves in each tree.
    The variance is computed as proposed by Hutter et al. in https://arxiv.org/pdf/1211.0906.pdf.
    :param leaf_per_sample: matrix with dimensions number_of_trees * number_of_samples. Stores the leaf each sample fell into for each tree.
    :param sample_means: list containing the mean of each sample.
    :param leaf_means: list of number_of_trees dictionaries. Each dictionary contains the mean for each leaf in a tree.
    :param leaf_vars: list of number_of_trees dictionaries. Each dictionary contains the variance for each leaf in a tree.
    :return: list containing the variance of each sample.
    """
    number_of_trees, number_of_samples = leaf_per_sample.shape
    mean_of_the_vars = np.zeros(number_of_samples)
    var_of_the_means = np.zeros(number_of_samples)
    sample_vars = np.zeros(number_of_samples)
    for sample_idx in range(number_of_samples):
        for tree_idx in range(number_of_trees):
            sample_leaf = leaf_per_sample[tree_idx, sample_idx]
            mean_of_the_vars[sample_idx] += leaf_vars[tree_idx][sample_leaf]/number_of_trees
            var_of_the_means[sample_idx] += (leaf_means[tree_idx][sample_leaf]**2)/number_of_trees

        var_of_the_means[sample_idx] = abs(var_of_the_means[sample_idx] - sample_means[sample_idx]**2)
        sample_vars[sample_idx] = mean_of_the_vars[sample_idx] + var_of_the_means[sample_idx]
        if sample_vars[sample_idx] == 0:
            sample_vars[sample_idx] = 0.00001

    return sample_vars


def get_samples_per_node(tree, leaf_per_sample):
    """
    Compute which samples passed through each node in a tree.
    :param tree: sklearn regression tree.
    :param leaf_per_sample: matrix with dimensions number_of_trees * number_of_samples. Stores the leaf each sample fell into for each tree.
    :return: list of lists. Each internal list contains which samples went through the node represented by the index in the outer list.
    """
    node_count = tree.tree_.node_count
    samples_per_node = [[] for i in range(node_count)]
    for sample_idx in range(len(leaf_per_sample)):
        leaf = leaf_per_sample[sample_idx]
        samples_per_node[leaf].append(sample_idx)

    parents = [-1]*node_count
    left_children = tree.tree_.children_left
    right_children = tree.tree_.children_right
    for node_idx in range(node_count):
        if left_children[node_idx] != -1:
            parents[left_children[node_idx]] = node_idx
        if right_children[node_idx] != -1:
            parents[right_children[node_idx]] = node_idx

    for node_idx in range(node_count-1, -1, -1):
        parent = parents[node_idx]
        if parent != -1:
            samples_per_node[parent] += samples_per_node[node_idx]
    return samples_per_node

def get_node_bounds(samples, data_array, threshold):
    """
    Compute the lower and upper bounds used to make a splitting decision at a tree node.
    :param samples: list containing the indices of all samples that went through the node.
    :param data_array: list containing the values of one parameter for all of the samples from the data.
    :param threshold: original threshold used to split the node.
    :return: lower and upper bound that were used to compute the split.
    """
    lower_bound = float('-inf')
    upper_bound = float('inf')
    for sample in samples:
        sample_value = data_array[sample]
        if sample_value <= threshold:
            lower_bound = max(lower_bound, data_array[sample])
        else:
            upper_bound = min(upper_bound, data_array[sample])

    return lower_bound, upper_bound

def transform_rf_using_uniform_splits(regression_models, data_array, param_space):
    """
    Change the splitting thresholds from a random forest regressor. Thresholds are changed from (upper_bound + lower_bound)/2
    to a uniformly sampled split in the range (lower_bound, upper_bound).
    This splitting approach was proposed by Hutter et al.: https://arxiv.org/pdf/1211.0906.pdf
    :param regression_models: sklearn random forest regressor. The model to adapt.
    :param data_array: dictionary containing the points evaluated so far.
    :param input_params: the input parameters of the regressor.
    :return: regressor with changed splitting rules.
    """
    input_params = param_space.get_input_parameters()
    bufferx = data_dictionary_to_tuple(data_array, input_params)
    leaf_per_sample = get_leaves_per_sample(bufferx, regression_models, param_space)
    preprocessed_data_array = preprocess_data_array(data_array, param_space, input_params)
    new_features = list(preprocessed_data_array.keys())
    for objective in regression_models:
        for tree_idx, tree in enumerate(regression_models[objective]):
            samples_per_node = get_samples_per_node(tree, leaf_per_sample[objective][tree_idx, :])

            left_children = tree.tree_.children_left
            right_children = tree.tree_.children_right
            for node_idx in range(tree.tree_.node_count):
                if left_children[node_idx] == right_children[node_idx]: # Its a leaf
                    continue
                feature = new_features[tree.tree_.feature[node_idx]]
                threshold = tree.tree_.threshold[node_idx]

                lower_bound, upper_bound = get_node_bounds(samples_per_node[node_idx], preprocessed_data_array[feature], threshold)
                new_split = stats.uniform.rvs(loc=lower_bound, scale=upper_bound - lower_bound)

                tree.tree_.threshold[node_idx] = new_split
    return regression_models
