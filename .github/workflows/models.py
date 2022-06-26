import numpy as np
import tensorflow
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

import dataprocessing

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


def make_my_model(pkt_data, series_len, np_seed, model_name, train=0.8, model_creator=None):
    global n_features
    n_features = len(pkt_data.columns)
    global series_length
    series_length = series_len

    X_train, X_test, y_train, y_test = custom_train_test_split(pkt_data, series_len, np_seed, train)

    dataprocessing.dump(dataprocessing.datasets_path, "X_train_{}".format(model_name), X_train)
    dataprocessing.dump(dataprocessing.datasets_path, "y_train_{}".format(model_name), y_train)
    dataprocessing.dump(dataprocessing.datasets_path, "X_test_{}".format(model_name), X_test)
    dataprocessing.dump(dataprocessing.datasets_path, "y_test_{}".format(model_name), y_test)

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
    tensorflow.keras.models.save_model(model, dataprocessing.modeles_path + model_name)

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
    dvs = dataprocessing.matrix_profiles_pre_processing(pkt_data, series_len, window, jump, np.argmin)
    lstm_series_length = series_len - window + 1

    dataprocessing.dump(dataprocessing.datasets_path, model_name, dvs)
    return simple_LSTM(dvs, lstm_series_length, np_seed, model_name)


def custom_train_test_split_series_pred(pkt_data, series_len, np_seed, train):
    n_features = len(pkt_data.columns)

    # make sure the number of packets is a multiple of series_len
    data_len = len(pkt_data)
    data_len -= data_len % series_len

    y = []
    X_grouped = []

    # save the data as sequence of length series_len
    i = 0

    while i < data_len - 2 * series_len:
        X_sequence = pkt_data.iloc[i: i + series_len].to_numpy().reshape(series_len, n_features)
        y_sequence = pkt_data.iloc[i + series_len: i + 2 * series_len].to_numpy().reshape(series_len, n_features)

        X_grouped.append(X_sequence)
        y.append(y_sequence)
        i += 1

    X_grouped = np.array(X_grouped)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X_grouped, y, test_size=1 - train, random_state=np_seed)

    return X_train, X_test, y_train, y_test


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


if __name__ == '__main__':
    model = tensorflow.keras.Sequential()
    model.add(layers.LSTM(units=4, input_shape=(20, 4), return_sequences=True))
    model.add(layers.LSTM(units=4, input_shape=(20, 4), return_sequences=True))
    model.add(layers.TimeDistributed(layers.Dense(4)))
    print(model.summary())
