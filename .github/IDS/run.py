import itertools

import pandas as pd

import data
import models
import models.TIRP as TIRP
import yaml


def get_models_folders_data_folders(train_config):
    params = yaml.load(train_config, Loader=yaml.FullLoader)
    binning_dict = params['binning']
    binning_part_in_folder = binning_dict.keys()
    data_versions = params['data_versions']
    data_folders = []
    models_folders = []
    for option in itertools.product(binning_part_in_folder, data_versions):
        binning_part = option[0]
        data_version = option[1]
        data_folders.append(binning_part + '_' + data_version)
        models_folders.append(binning_part + '-' + data_version)

    return models_folders, data_folders, binning_dict, params


def train_RF(RF_train_config_file_path):
    with open(RF_train_config_file_path, mode='r') as train_config:
        models_folders, data_folders, binning_dict, params = get_models_folders_data_folders(train_config)
        zipped = itertools.product(models_folders, data_folders)
        for folder_pair in zipped:
            models_folder = folder_pair[0]
            data_folder = folder_pair[1]
            binning_version = (data_folder.split(sep='_'))[0]
            models.models.make_classifier(models_folder=models_folder, data_folder=data_folder,
                                          binning=binning_dict[binning_version], params=params, RF_only=True)


def train_OCSVM(OCSVM_train_config_file_path):
    with open(OCSVM_train_config_file_path, mode='r') as train_config:
        models_folders, data_folders, binning_dict, params = get_models_folders_data_folders(train_config)
        zipped = itertools.product(models_folders, data_folders)
        for folder_pair in zipped:
            models_folder = folder_pair[0]
            data_folder = folder_pair[1]
            binning_version = (data_folder.split(sep='_'))[0]
            models.models.make_classifier(models_folder=models_folder, data_folder=data_folder,
                                          binning=binning_dict[binning_version], params=params, RF_only=True)


def train_automaton(Automaton_train_config_file_path):
    names = {"KMeans": data.k_means_binning, "EqualFreq": data.equal_frequency_discretization,
             "EqualWidth": data.equal_width_discretization}
    train_df = pd.DataFrame()
    with open(Automaton_train_config_file_path) as train_config:
        params = yaml.load(train_config, Loader=yaml.FullLoader)
        binning_methods = params['binning']
        number_bins = params['number_bins']
        to_scale = params['to_scale']

        binners_bins_product = itertools.product(binning_methods, number_bins)
        options = itertools.product(to_scale, binners_bins_product)
        for option in options:
            scale = option[0]
            binner = scale[1][0]
            bins = scale[1][1]
            if binner is None:
                dfa = models.automaton.make_automaton(data.to_bin, train_df, None, None, scale)
            else:
                dfa = models.automaton.make_automaton(data.to_bin, train_df, names[binner], bins, scale)
            data.dump(data.automaton_path, '{}_{}_scale_{}.sav'.format(binner, bins, scale), dfa)


def make_input_for_KL(TIRP_config_file_path):
    pkt_df = data.load(data.datasets_path, "modbus")
    IP = data.plc
    # consider only response packets from the PLC.
    plc_df = pkt_df.loc[(pkt_df['src_ip'] == IP) & (pkt_df['dst_ip'] == IP)]
    stats_dict = data.get_plcs_values_statistics(plc_df, 5, to_df=False)
    with open(TIRP_config_file_path, mode='r') as train_config:
        params = yaml.load(train_config, Loader=yaml.FullLoader)
        discovery_params = params['TIRP_discovery_params']
        binning = discovery_params['binning']
        binning_methods = {'KMeans': TIRP.k_means_binning, 'EqualFreq': TIRP.equal_frequency_discretization,
                           'EqualWidth': TIRP.equal_width_discretization}
        number_of_bins = discovery_params['number_of_bins']
        windows_dict = discovery_params['windows_sizes']
        start = windows_dict['start']
        end = windows_dict['end']
        jump = windows_dict['jump']
        windows = range(start, end, jump)
        bins_window_options = itertools.product(number_of_bins, windows)
        options = itertools.product(binning, bins_window_options)
        for option in options:
            b = binning_methods[option[0]]
            k = option[1][0]
            w = option[1][1]
            TIRP.discover(plc_df, b, k, w, consider_last=True, stats_dict=stats_dict)
            TIRP.make_input(plc_df, b, k, w, stats_dict, consider_last=True)


def train_RF_from_KL(TIRP_config_file_path):
    with open(TIRP_config_file_path, mode='r') as train_config:
        params = yaml.load(train_config, Loader=yaml.FullLoader)
        RF_params = params['RF']
        parameters_combinations = itertools.product(RF_params['criterion'], RF_params['max_features'])
        parameters_combinations = itertools.product(RF_params['n_estimators'], parameters_combinations)
        for combination in parameters_combinations:
            estimators = combination[0]
            criterion = combination[1][0]
            max_features = combination[1][1]
