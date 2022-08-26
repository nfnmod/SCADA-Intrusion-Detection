import itertools
import os
import pickle

import keras
import numpy as np
import tensorflow
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDOneClassSVM
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from tensorflow.keras import layers

import data

# TODO: create train data sets for classifiers if needed- before training.


n_features = -1
series_length = -1


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


def simple_LSTM(pkt_data, series_len, np_seed, model_name, train=0.8):
    return make_my_model(pkt_data, series_len, np_seed, model_name, train, build_LSTM)


def build_One_Class_SVM(kernel, nu):
    svm = OneClassSVM(kernel=kernel, nu=nu)
    return svm


def build_SGD(nu):
    sgd = SGDOneClassSVM(nu=nu)
    return sgd


# train classifier with models from the first folder and train sets from the other folder.
# Classifiers are trained with benign data and tested with anomalies.
# MAKE SURE THE DATA FOLDER HAS ALL THE TRAIN SETS.
def make_classifier(models_folder, data_folder, binning):
    for model_folder in os.listdir(data.modeles_path + '\\' + models_folder):
        model_path = data.modeles_path + "\\" + models_folder + "\\" + model_folder
        model = keras.models.load_model(model_path)
        for bins in range(5, 11):
            x_train_path = data.datasets_path + data_folder + '\\X_train_' + model_folder + '_{}_{}'.format(binning, bins)
            y_train_path = data.datasets_path + data_folder + '\\y_train_' + model_folder + '_{}_{}'.format(binning, bins)
            with open(x_train_path, 'rb') as x_train_f:
                x_train = pickle.load(x_train_f)
            with open(y_train_path, 'rb') as y_train_f:
                y_train = pickle.load(y_train_f)
            post_lstm_classifier_One_Class_SVM(model, x_train, y_train, model_folder + '_OCSVM')


def post_lstm_classifier_One_Class_SVM(lstm_model, x_train, y_train, model_name):
    """

    :param lstm_model: an LSTM model fitted on normal traffic
    :param x_train: the data the model was trained on
    :param y_train: the data the model was trained on
    :param model_name: for saving files
    :return: a classifier for the differences. (try to tell if the packet is anomalous according to the deviation of the LSTM)
    # from the reality.
    """

    params_dict = dict()
    params_dict['kernel'] = ['poly', 'rbf', 'sigmoid']
    params_dict['nu'] = [0.05, 0.1, 0.15, 0.2]

    # predict and get the difference from the truth.
    # this is the training data for the SVM.
    # when testing, inject anomalies into x_test and the corresponding y_test entries. the LSTM prediction should differ in a
    # different way than it does for benign packets.
    pred = lstm_model.predict(x_train)
    diff_x_train = np.abs(pred - y_train)
    data.dump(data.datasets_path, "X_train_{}".format(model_name), diff_x_train)

    for kernel in params_dict['kernel']:
        k = kernel
        for nu in params_dict['nu']:
            n = nu
            model = build_One_Class_SVM(k, n)
            model.fit(diff_x_train)
            tensorflow.keras.models.save_model(model,
                                               data.modeles_path + model_name + 'nu_{}'.format(n) + 'kernel_{}'.format(
                                                   k))


def post_lstm_classifier_Random_Forest(lstm_model, x_train, y_train, model_name):
    params_dict = dict()
    params_dict['n_estimators'] = [50, 100]
    params_dict['criterion'] = ['gini', 'entropy', 'log_loss']
    params_dict['max_features'] = ['sqrt', 'log2', None]

    # predict and get the difference from the truth.
    # this is the training data for the SVM
    pred = lstm_model.predict(x_train)
    diff_x_train = np.abs(pred - y_train)
    data.dump(data.datasets_path, "X_train_{}".format(model_name), diff_x_train)

    parameters_combinations = itertools.product(params_dict['criterion'], params_dict['max_features'])
    parameters_combinations = itertools.product(params_dict['n_estimators'], parameters_combinations)

    for combination in parameters_combinations:
        estimators = combination[0]
        criterion = combination[1][0]
        max_features = combination[1][1]
        model = RandomForestClassifier(n_estimators=estimators, criterion=criterion, max_features=max_features)
        model.fit(diff_x_train, np.zeros((-1, len(x_train))))
        tensorflow.keras.models.save_model(model, data.modeles_path + model_name + 'estimators_{}_'.format(estimators) + 'criterion{}_'.format(
                                                   criterion) + 'features_{}'.format(max_features))


def make_my_model(pkt_data, series_len, np_seed, model_name, train=0.8, model_creator=None):
    global n_features
    n_features = len(pkt_data.columns)
    global series_length
    series_length = series_len

    X_train, X_test, y_train, y_test = custom_train_test_split(pkt_data, series_len, np_seed, train)

    data.dump(data.datasets_path, "X_train_{}".format(model_name), X_train)
    data.dump(data.datasets_path, "y_train_{}".format(model_name), y_train)
    data.dump(data.datasets_path, "X_test_{}".format(model_name), X_test)
    data.dump(data.datasets_path, "y_test_{}".format(model_name), y_test)

    kf = KFold(n_splits=10, random_state=np_seed, shuffle=True)

    params_dict = dict()
    params_dict['epochs'] = np.linspace(3, 15, num=13, dtype=np.int)
    params_dict['batch_size'] = [32, 48, 64]

    estimator = KerasRegressor(build_fn=model_creator)
    search = GridSearchCV(estimator=estimator, param_grid=params_dict, cv=kf, verbose=3,
                          scoring='neg_mean_squared_error')

    print("fitting the model")
    best_model = search.fit(X_train, y_train)
    best_params = best_model.best_params_

    model = model_creator(best_params['epochs'], best_params['batch_size'])
    model.fit(X_train, y_train)
    tensorflow.keras.models.save_model(model, data.modeles_path + model_name)

    print('Best Score: %s' % best_model.best_score_)
    print('Best Hyper parameters: %s' % best_model.best_params_)

    return best_model


def build_LSTM(epochs, batch_size):
    model = tensorflow.keras.Sequential()
    model.add(layers.LSTM(units=n_features, input_shape=(series_length, n_features)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(n_features, activation='relu'))
    model.compile(loss=tensorflow.keras.losses.MeanSquaredError(), metrics=["mean_squared_error", "accuracy"],
                  optimizer='adam', )

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
    X_train, X_test, y_train, y_test = train_test_split(X_grouped, y, test_size=1 - train, random_state=np_seed)

    return X_train, X_test, y_train, y_test
