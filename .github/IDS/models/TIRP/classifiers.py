"""
This class is responsible for using the functionality provided by the output class in order to
train classifiers.
"""
import itertools

import tensorflow
from sklearn.ensemble import RandomForestClassifier

models_path = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\KL_RF\\'


def train_classifier(train_df, model_name):
    params_dict = dict()
    params_dict['n_estimators'] = [50, 100]
    params_dict['criterion'] = ['gini', 'entropy']
    params_dict['max_features'] = ['sqrt', 'log2', None]

    parameters_combinations = itertools.product(params_dict['criterion'], params_dict['max_features'])
    parameters_combinations = itertools.product(params_dict['n_estimators'], parameters_combinations)

    for combination in parameters_combinations:
        estimators = combination[0]
        criterion = combination[1][0]
        max_features = combination[1][1]
        model = RandomForestClassifier(n_estimators=estimators, criterion=criterion, max_features=max_features)
        model.fit(train_df.iloc[:, :-1], train_df.iloc[:, -1])
        tensorflow.keras.models.save_model(model, models_path + model_name + '_estimators_{}_'.format(
            estimators) + 'criterion_{}_'.format(
            criterion) + 'features_{}'.format(max_features))
