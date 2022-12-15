import itertools
import os
import pickle
import time
from pathlib import Path

import keras
import numpy as np
import tensorflow
import yaml
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDOneClassSVM
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from tensorflow.keras import layers

import data

n_features = -1
series_length = -1

logs = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\log files\\'
LSTM_based_OCSVM_log = logs + 'LSTM based OCSVM.txt'
LSTM_based_RF_log = logs + 'LSTM based RF.txt'
SCADA_base = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA'
LSTM_train_log = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\log files\\LSTM train.txt'


def build_LSTM_series_prediction(epochs, batch_size):
    model = tensorflow.keras.Sequential()
    model.add(layers.LSTM(units=n_features, input_shape=(series_length, n_features), return_sequences=True))
    model.add(layers.LSTM(units=n_features, input_shape=(series_length, n_features), return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(units=n_features)))
    model.compile(loss=tensorflow.keras.losses.MeanSquaredError(), metrics=["mean_squared_error", "accuracy"],
                  optimizer='adam', )
    return model


def build_RNN(epochs, batch_size):
    model = tensorflow.keras.Sequential()
    # model.add(layers.SimpleRNN(units=n_features, input_shape=(series_length, n_features), return_sequences=True))
    model.add(layers.SimpleRNN(units=n_features, input_shape=(series_length, n_features)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(n_features, activation='relu'))
    model.compile(loss=tensorflow.keras.losses.MeanSquaredError(), metrics=["mean_squared_error", "accuracy"],
                  optimizer='adam', )

    return model


def simple_RNN(pkt_data, series_len, np_seed, model_name, train=0.8):
    return make_my_model(pkt_data, series_len, np_seed, model_name, train, build_RNN)


def predict_series_LSTM(pkt_data, series_len, np_seed, model_name, train=0.8):
    return make_my_model(pkt_data, series_len, np_seed, model_name, train, build_LSTM_series_prediction)


def simple_LSTM(pkt_data, series_len, np_seed, model_name, train=0.8, models_path=data.modeles_path, data_path=data.datasets_path):
    return make_my_model(pkt_data, series_len, np_seed, model_name, train, build_LSTM, dump_model=models_path, dump_df=data_path)


def build_One_Class_SVM(kernel, nu):
    svm = OneClassSVM(kernel=kernel, nu=nu)
    return svm


def build_SGD(nu):
    sgd = SGDOneClassSVM(nu=nu)
    return sgd


# train classifier with models from the first folder and train sets from the other folder.
# Classifiers are trained with benign data and tested with anomalies.
# MAKE SURE THE DATA FOLDER HAS ALL THE TRAIN SETS.
def make_classifier(models_folder, data_folder, params, RF_only=False, OCSVM_only=False):
    # models folder described the data version and binning method.
    for model_folder in os.listdir(data.modeles_path + '\\' + models_folder):
        # model folder is the specific LSTM model.
        model_path = data.modeles_path + "\\" + models_folder + "\\" + model_folder
        model = keras.models.load_model(model_path)

        x_train_path = data.datasets_path + data_folder + '\\X_train_' + model_folder
        y_train_path = data.datasets_path + data_folder + '\\y_train_' + model_folder
        with open(x_train_path, 'rb') as x_train_f:
            x_train = pickle.load(x_train_f)
        with open(y_train_path, 'rb') as y_train_f:
            y_train = pickle.load(y_train_f)
        if not RF_only and not OCSVM_only:
            post_lstm_classifier_One_Class_SVM(model, x_train, y_train, model_folder + '_OCSVM', params,
                                               models_folder)
            post_lstm_classifier_Random_Forest(model, x_train, y_train, model_folder + '_RF', params, models_folder)
        elif OCSVM_only:
            post_lstm_classifier_One_Class_SVM(model, x_train, y_train, model_folder + '_OCSVM', params,
                                               models_folder)
        else:
            post_lstm_classifier_Random_Forest(model, x_train, y_train, model_folder + '_RF', params, models_folder)


def post_lstm_classifier_One_Class_SVM(lstm_model, x_train, y_train, model_name, params, models_folder):
    """

    :param models_folder: name of models folder of LSTMs, used for convenient saving of classifiers.
    :param params: config params for training
    :param lstm_model: an LSTM model fitted on normal traffic
    :param x_train: the data the model was trained on
    :param y_train: the data the model was trained on
    :param model_name: for saving files
    :return: a classifier for the differences. (try to tell if the packet is anomalous according to the deviation of the LSTM)
    # from the reality.
    """
    config_params_dict = params['params_dict']
    params_dict = dict()
    params_dict['kernel'] = config_params_dict['kernel']
    params_dict['nu'] = config_params_dict['nu']

    # predict and get the difference from the truth.
    # this is the training data for the SVM.
    # when testing, inject anomalies into x_test and the corresponding y_test entries. the LSTM prediction should differ in a
    # different way than it does for benign packets.
    pred = lstm_model.predict(x_train)
    diff_x_train = np.abs(pred - y_train)
    # dirs for the datasets.
    diff_path = SCADA_base + '\\OCSVM datasets\\OCSVM_diff_{}'.format(models_folder)
    raw_path = SCADA_base + '\\OCSVM datasets\\OCSVM_{}'.format(models_folder)
    if not os.path.exists(diff_path):
        Path(diff_path).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(raw_path):
        Path(raw_path).mkdir(parents=True, exist_ok=True)
    data.dump(diff_path, "diff_X_train_{}".format(model_name),
              diff_x_train)
    data.dump(raw_path, "X_train_{}".format(model_name), pred)

    binning_version = models_folder.split(sep='_', maxsplit=1)
    binning = binning_version[0]
    version = binning_version[1]
    number_of_bins = model_name.split(sep='_')[-2]
    for kernel in params_dict['kernel']:
        k = kernel
        for nu in params_dict['nu']:
            n = nu
            model = build_One_Class_SVM(k, n)
            with open(LSTM_based_OCSVM_log, mode='a') as log:
                log.write('Training LSTM (diff from pred) based OCSVM with:\n')
                log.write(
                    'data version: {}, binning: {}, number of bins: {}\n'.format(version, binning, number_of_bins))
                log.write('kernel:{}, nu:{}\n'.format(k, n))
                start = time.time()
                model.fit(diff_x_train)
                end = time.time()
                log.write('Trained, time elapsed:{}\n'.format(end - start))
                dirc = SCADA_base + '\\SVMs\\' + 'diff_' + models_folder
                p = 'diff_' + model_name + '_nu_{}_'.format(
                    n) + 'kernel_{}.sav'.format(
                    k)
                if not os.path.exists(dirc):
                    Path(dirc).mkdir(exist_ok=True, parents=True)
                data.dump(dirc, p, model)
                log.write('Training LSTM (pred) based OCSVM with:')
                log.write(
                    'data version: {}, binning: {}, number of bins: {}\n'.format(version, binning, number_of_bins))
                log.write('kernel:{}, nu:{}\n'.format(k, n))
                model_raw = build_One_Class_SVM(k, n)
                start = time.time()
                model_raw.fit(pred)
                end = time.time()
                dirc = SCADA_base + '\\SVMs\\' + models_folder
                p = model_name + '_nu_{}_'.format(
                    n) + 'kernel_{}.sav'.format(
                    k)
                if not os.path.exists(dirc):
                    Path(dirc).mkdir(exist_ok=True, parents=True)
                data.dump(dirc, p, model_raw)
                log.write('Trained, time elapsed:{}\n'.format(end - start))


def post_lstm_classifier_Random_Forest(lstm_model, x_train, y_train, model_name, params, models_folder):
    config_params_dict = params['params_dict']
    params_dict = dict()
    params_dict['n_estimators'] = config_params_dict['n_estimators']
    params_dict['criterion'] = config_params_dict['criterion']
    params_dict['max_features'] = config_params_dict['max_features']

    # predict and get the difference from the truth.
    # this is the training data for the SVM
    pred = lstm_model.predict(x_train)
    diff_x_train = np.abs(pred - y_train)
    diff_path = SCADA_base + '\\RF datasets\\RF_diff_{}'.format(models_folder)
    raw_path = SCADA_base + '\\RF datasets\\RF_{}'.format(models_folder)
    if not os.path.exists(diff_path):
        Path(diff_path).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(raw_path):
        Path(raw_path).mkdir(parents=True, exist_ok=True)
    data.dump(diff_path, "diff_X_train_{}".format(model_name),
              diff_x_train)
    data.dump(raw_path, "X_train_{}".format(model_name), pred)

    parameters_combinations = itertools.product(params_dict['criterion'], params_dict['max_features'])
    parameters_combinations = itertools.product(params_dict['n_estimators'], parameters_combinations)

    binning_version = models_folder.split(sep='_', maxsplit=1)
    binning = binning_version[0]
    version = binning_version[1]
    number_of_bins = model_name.split(sep='_')[-2]
    for combination in parameters_combinations:
        estimators = combination[0]
        criterion = combination[1][0]
        max_features = combination[1][1]
        with open(LSTM_based_RF_log, mode='a') as log:
            model = RandomForestClassifier(n_estimators=estimators, criterion=criterion, max_features=max_features)
            log.write('Training LSTM (diff from pred) based RF with: \n')
            log.write('data version: {}, binning: {}, number of bins: {}\n'.format(version, binning, number_of_bins))
            log.write('estimators:{}, criterion:{}, max_features:{}\n'.format(estimators, criterion, max_features))
            start = time.time()
            model.fit(diff_x_train, np.zeros(len(x_train)))
            end = time.time()
            log.write('Trained, time elapsed:{}\n'.format(end - start))
            dirc = SCADA_base + '\\RFs\\' + 'diff_' + models_folder
            p = 'diff_' + model_name + '_estimators_{}_'.format(
                estimators) + 'criterion_{}_'.format(
                criterion) + 'features_{}.sav'.format(max_features)
            if not os.path.exists(dirc):
                Path(dirc).mkdir(exist_ok=True, parents=True)
            data.dump(dirc, p, model)
            model_raw = RandomForestClassifier(n_estimators=estimators, criterion=criterion, max_features=max_features)
            log.write('Training LSTM (pred) based RF with:\n')
            log.write('data version: {}, binning: {}, number of bins: {}\n'.format(version, binning, number_of_bins))
            log.write('estimators:{}, criterion:{}, max_features:{}\n'.format(estimators, criterion, max_features))
            start = time.time()
            model_raw.fit(pred, np.zeros(len(pred)))
            end = time.time()
            log.write('Trained, time elapsed:{}\n'.format(end - start))
            dirc = SCADA_base + '\\RFs\\' + models_folder
            p = model_name + '_estimators_{}_'.format(
                estimators) + 'criterion_{}_'.format(
                criterion) + 'features_{}.sav'.format(max_features)
            if not os.path.exists(dirc):
                Path(dirc).mkdir(exist_ok=True, parents=True)
            data.dump(dirc, p, model_raw)


def make_my_model(pkt_data, series_len, np_seed, model_name, train=0.8, model_creator=None, dump_df=data.datasets_path,
                  dump_model=data.modeles_path):
    global n_features
    n_features = len(pkt_data.columns)
    global series_length
    series_length = series_len

    X_train, X_test, y_train, y_test = custom_train_test_split(pkt_data, series_len, np_seed, train)

    data.dump(dump_df, "X_train_{}".format(model_name), X_train)
    data.dump(dump_df, "y_train_{}".format(model_name), y_train)
    # data.dump(dump_df, "X_test_{}".format(model_name), X_test)
    # data.dump(dump_df, "y_test_{}".format(model_name), y_test)

    kf = KFold(n_splits=10, random_state=np_seed, shuffle=True)

    early_stopping = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)

    params_dict = dict()
    params_dict['epochs'] = [20, 30, 40]
    params_dict['batch_size'] = [32, 64, 128]
    params_dict['callbacks'] = [[early_stopping]]

    estimator = KerasRegressor(build_fn=model_creator)
    search = GridSearchCV(estimator=estimator, param_grid=params_dict, cv=kf, scoring='neg_mean_squared_error')

    print("fitting the model")
    start = time.time()
    best_model = search.fit(X_train, y_train)
    end = time.time()
    with open(LSTM_train_log, mode='a') as train_log:
        train_log.write(model_name + ': {} seconds'.format(end - start))
    best_params = best_model.best_params_

    model = model_creator(best_params['epochs'], best_params['batch_size'], best_params['callbacks'])
    model.fit(X_train, y_train, best_params['callbacks'])
    tensorflow.keras.models.save_model(model, dump_model + '\\' + model_name)

    print('Best Score: %s' % best_model.best_score_)
    print('Best Hyper parameters: %s' % best_model.best_params_)

    return best_model


def grid_search_train(pkt_data, series_len, np_seed, model_name, train=0.8):
    global n_features
    n_features = len(pkt_data.columns)
    global series_length
    series_length = series_len

    X_train, X_test, y_train, y_test = custom_train_test_split(pkt_data, series_len, np_seed, train)

    kf = KFold(n_splits=10, random_state=np_seed, shuffle=True)

    params_dict = dict()
    params_dict['epochs'] = np.linspace(3, 15, num=13, dtype=np.int)
    params_dict['batch_size'] = [32, 48, 64]

    for train, test in kf.split(X_train, y_train):
        X_split_train, X_split_test = X_train[train], X_train[test]
        y_split_train, y_split_test = y_train[train], y_train[test]
        for epochs in params_dict['epochs']:
            for batch in params_dict['batch_size']:
                model = build_LSTM(epochs, batch)
                with open(LSTM_train_log, mode='a') as log:
                    log.write('training with: epochs = {}, batch size = {}\n'.format(epochs, batch))
                start = time.time()
                model.fit(X_split_train, y_split_train)
                end = time.time()
                with open(LSTM_train_log, mode='a') as log:
                    log.write('trained, time elapsed: {}\n'.format(end - start))


def build_LSTM(epochs, batch_size, callbacks):
    model = tensorflow.keras.Sequential()
    model.add(layers.LSTM(units=n_features, input_shape=(series_length, n_features)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(n_features, activation='relu'))
    model.compile(loss=tensorflow.keras.losses.MeanSquaredError(), metrics=["mean_squared_error"],
                  optimizer='adam')

    return model


def matrix_profiles_LSTM(pkt_data, series_len, window, jump, np_seed, model_name):
    dvs = data.matrix_profiles_pre_processing(pkt_data, series_len, window, jump, np.argmin)
    lstm_series_length = series_len - window + 1

    data.dump(data.datasets_path, model_name, dvs)
    return simple_LSTM(dvs, lstm_series_length, np_seed, model_name)


def custom_train_test_split(pkt_data, series_len, np_seed, train=0.8):
    n_features = len(pkt_data.columns)

    # make sure the number of packets is a multiple of series_len
    data_len = len(pkt_data)
    data_len -= data_len % series_len

    y = []
    X_grouped = []

    # save the data as sequence of length series_len
    i = 0

    while i < data_len - series_len:
        X_sequence = pkt_data.iloc[i:i + series_len, 0: n_features].to_numpy().reshape(series_len, n_features)
        y_sequence = pkt_data.iloc[i + series_len, 0: n_features].to_numpy().reshape(1, n_features)[0]

        X_grouped.append(X_sequence)
        y.append(y_sequence)
        i += 1

    X_grouped = np.array(X_grouped)
    y = np.array(y)
    X_grouped = np.asarray(X_grouped).astype(np.float32)
    y = np.asarray(y).astype(np.float32)
    if train == 0 or train == 1:
        return X_grouped, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_grouped, y, test_size=1 - train, random_state=np_seed)
        return X_train, X_test, y_train, y_test


def train_LSTM_many_PLCs(datasets_base_path, train_config):
    # for each configuration of LSTM training:
    #   train LSTM for each group.
    #   save models.
    with open(train_config, mode='r') as config:
        LSTM_params = yaml.load(config, Loader=yaml.FullLoader)
    versions = LSTM_params['version_name']
    bins = LSTM_params['bins']
    binning_methods = LSTM_params['binning_method']
    for version in versions:
        name = version['name']
        description = versions['description']
        for number_of_bins in bins:
            for binning_method in binning_methods:
                for dataset in os.listdir(datasets_base_path):
                    group_id = dataset.split(sep='_')[-1]
                    with open(datasets_base_path + '\\' + dataset, mode='wb') as df_path:
                        raw_df = pickle.load(df_path)
                    processed = data.process(raw_df, name, number_of_bins, binning_method)
                    model_name = description + '{}_{}'.format(binning_method, number_of_bins)
                    df_p = data.datasets_path + '\\{}_{}_{}'.format(group_id, binning_method, name)
                    model_p = data.modeles_path + '\\{}_{}_{}'.format(group_id, binning_method, name)
                    if not os.path.exists(df_p):
                        Path(df_p).mkdir(exist_ok=True, parents=True)
                    if not os.path.exists(model_p):
                        Path(model_p).mkdir(exist_ok=True, parents=True)
                    make_my_model(processed, 20, 42, model_name, train=0.8, model_creator=simple_LSTM, dump_df=df_p, dump_model=model_p)

