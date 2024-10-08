import csv
import itertools
import os
import pickle
import re
import time
from pathlib import Path

import keras.models
import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.spatial import distance
from scipy.stats import stats
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score, r2_score, auc, precision_score, \
    recall_score, mean_squared_error, confusion_matrix
from sklearn.svm import OneClassSVM

import data
import models
import models.TIRP as TIRP
from data import squeeze, find_frequent_transitions_sequences
from data.injections import inject_to_sub_group
from models.models import LSTM_train_log

KL_base = data.datasets_path + "\\KL\\"
KL_RF_base = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\KL RF'
FSTM_data = data.datasets_path + '\\FSTM'
HTM_base = "C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\HTM\\"
logs = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\log files\\'
test_sets_base_folder = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\test sets'
KL_based_OCSVM_log = logs + 'KarmaLego based OCSVM.txt'
KL_LSTM_log = logs + 'KL-LSTM.txt'
lstm_test_log = logs + 'LSTM test.txt'
DFA_log = logs + 'DFA.txt'
FSTM_train_log = logs + 'FSTM train.txt'
KL_output_base = "C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\test sets\\KL\\KL out"
TIRPs_base = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\KL TIRPS'
LSTM_classifiers_classifications = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\LSTM_classifications'
group_df_base = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\datasets\\group'
binners_base = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\binners'
scalers_base = '//sise//home//zaslavsm//SCADA//scalers'
DFA_regs = ['30', '120', '15']
KL_OCSVM_datasets = models.SCADA_base + '\\KL_OCSVM datasets'
KL_OCSVM_base = models.SCADA_base + '\\KL_OCSVM'
KL_test_sets_base = test_sets_base_folder + "//KLSTM//events"

excel_cols_old = {'HTM type', 'LSTM type', 'mix', 'data version', 'binning', '# bins', 'nu', 'kernel', '# estimators',
                  'criterion', 'max features',
                  'ON bits', 'SDR size', 'numOfActiveColumnsPerInhArea', 'potential Pct', 'synPermConnected',
                  'synPermActiveInc', 'synPermInactiveDec', 'boostStrength', 'cellsPerColumn', 'newSynapseCount',
                  'initialPerm', 'permanenceInc', 'permanenceDec', 'maxSynapsesPerSegment', 'maxSegmentsPerCell',
                  'minThreshold', 'activationThreshold', 'window size', 'KL epsilon', 'minimal VS', 'max gap',
                  'injection length', 'step over', 'percentage', 'precision', 'recall', 'f1'}
"""excel_cols = ['algorithm', 'data version', 'binning', '# bins', '# std count', 'window size', 'likelihood_threshold',
              'ON bits', 'SDR size', 'numOfActiveColumnsPerInhArea', 'potential Pct', 'synPermConnected',
              'synPermActiveInc', 'synPermInactiveDec', 'boostStrength', 'cellsPerColumn', 'newSynapseCount',
              'initialPerm', 'permanenceInc', 'permanenceDec', 'maxSynapsesPerSegment', 'maxSegmentsPerCell',
              'minThreshold', 'activationThreshold', 'KL epsilon', 'minimal K', 'max gap',
              'injection length', 'percentage', 'precision', 'recall', 'auc', 'f1', 'prc']"""
excel_cols = ['group', 'injection type', 'algorithm', 'data version', 'binning', '# bins', '# std count', 'kernel',
              'nu', 'likelihood_threshold', 'window size',
              'ON bits', 'SDR size', 'numOfActiveColumnsPerInhArea', 'potential Pct', 'synPermConnected',
              'synPermActiveInc', 'synPermInactiveDec', 'boostStrength', 'cellsPerColumn',
              'initialPerm', 'permanenceInc', 'permanenceDec', 'maxSynapsesPerSegment', 'maxSegmentsPerCell',
              'minThreshold', 'activationThreshold', 'KL epsilon', 'minimal HS', 'max gap',
              'injection length', 'percentage', 'precision', 'recall', 'f1']

RF_cols = {'data version', 'binning', '# bins', '# estimators', 'criterion', 'max features',
           'injection length', 'step over', 'percentage', 'precision', 'recall', 'f1'}
OCSVM_cols = ['group', 'injection type', 'data version', 'binning', '# bins', '# std count', 'window size', 'nu',
              'kernel',
              'injection length', 'step over', 'percentage', 'precision', 'recall', 'f1']

DFA_cols = {'group', 'injection type', 'binning', '# bins', '# std count', 'injection length', 'step over',
            'percentage', 'precision',
            'recall', 'f1'}

LSTM_detection_cols = ['group', 'injection type', 'data version', 'binning', '# bins', '# std count', 'window size',
                       'injection length',
                       'step over', 'percentage',
                       'precision',
                       'recall', 'f1']

KL_based_RF_cols = {'binning', '# bins', 'window size', 'KL epsilon', 'minimal K', 'max gap',
                    'injection length', 'step over', 'percentage', 'precision', 'recall',
                    'f1'}
# xl_path = '//sise//home//zaslavsm//SCADA//excel//DFA sheets.xlsx'
xl_path2 = 'C:\\Users\\michaelzaslavski\\OneDrive\\Desktop\\SCADA\\excel\\LSTM v3_2 sheets.xlsx'
xl_path3 = '//sise//home//zaslavsm//SCADA//excel//LSTM v2 sheets.xlsx'
xl_path4 = '//sise//home//zaslavsm//SCADA//excel//LSTM v1_1 sheets.xlsx'
paths_dict = {'v1_1': xl_path4, 'v2': xl_path3, 'v3_2': xl_path2}

HTM_cols = ['group', 'injection type', 'data version', 'binning', '# bins', '# std count', 'likelihood_threshold',
            'window size',
            'ON bits',
            'SDR size', 'columnCount',
            'numActiveColumnsPerInhArea', 'potentialPct',
            'synPermConnected',
            'synPermActiveInc', 'synPermInactiveDec',
            'boostStrength', 'cellsPerColumn', 'initialPerm', 'permanenceInc', 'permanenceDec',
            'maxSynapsesPerSegment',
            'maxSegmentsPerCell', 'minThreshold', 'activationThreshold', 'injection length', 'step over', 'percentage',
            'precision',
            'recall', 'f1']

window_sizes = [400, 600, 800, 1000]
p_values = [0.01, 0.03, 0.05]

nums_std = [0, 1, 2, 3]
num_stds_count = [0, 1, 2, 3]

spearman_groups = ['spearman_0', 'spearman_1']
pearson_groups = ['pearson_0', 'pearson_1']
k_means_groups = ['k_means_0', 'k_means_1']

spearman_groups_ips = [['132.72.249.42', '132.72.248.211', '132.72.155.245'], ['132.72.35.161', '132.72.75.40']]
pearson_groups_ips = [['132.72.75.40', '132.72.248.211', '132.72.155.245'], ['132.72.249.42', '132.72.35.161']]
k_means_groups_ips = [['132.72.75.40', '132.72.248.211', '132.72.35.161'], ['132.72.249.42', '132.72.155.245']]

effected_plcs_data_tuples = [(spearman_groups_ips[0], spearman_groups[0]), (spearman_groups_ips[1], spearman_groups[1]),
                             (pearson_groups_ips[0], pearson_groups[0]), (pearson_groups_ips[1], pearson_groups[1]),
                             (k_means_groups_ips[0], k_means_groups[0]), (k_means_groups_ips[1], k_means_groups[1])]
SPEARMAN = 'spearman'
PEARSON = 'pearson'
K_MEANS = 'k_means'
SINGLE_PLC = 'single_plc'
ALL_PLCS = 'all_plcs'
SPLIT_TYPES = [SINGLE_PLC, ALL_PLCS, SPEARMAN, PEARSON, K_MEANS]

SPLIT_TYPE_TO_GROUP_POOL = {SPEARMAN: spearman_groups,
                            PEARSON: pearson_groups,
                            K_MEANS: k_means_groups,
                            SINGLE_PLC: ['single_plc_' + active_ip for active_ip in data.active_ips],
                            ALL_PLCS: [ALL_PLCS]
                            }

injection_type_to_effected_plcs_suffixes = {SPEARMAN: spearman_groups,
                                            PEARSON: pearson_groups,
                                            K_MEANS: k_means_groups,
                                            SINGLE_PLC: [active_ip for active_ip in data.active_ips],
                                            ALL_PLCS: [None]}

best_cols = DFA_cols.copy()
lim = 0.2

xl_base = "C:\\Users\\michaelzaslavski\\OneDrive\\Desktop\\SCADA\\excel\\"
xl_path = 'C:\\Users\\michaelzaslavski\\OneDrive\\Desktop\\SCADA\\excel\\DFA sheets.xlsx'

xl_labels_path = 'C:\\Users\\michaelzaslavski\\OneDrive\\Desktop\\SCADA\\excel\\labels.xlsx'

test_LSTM_OCSVM_log = 'C:\\Users\\michaelzaslavski\\OneDrive\\Desktop\\SCADA\\log files\\test LSTM_OCSVM.txt'
test_DFA_log = 'C:\\Users\\michaelzaslavski\\OneDrive\\Desktop\\SCADA\\log files\\test DFA.txt'
test_KL_RF_log = 'C:\\Users\\michaelzaslavski\\OneDrive\\Desktop\\SCADA\\log files\\test KL-RF.txt'
test_LSTM_STD_log = 'C:\\Users\\michaelzaslavski\\OneDrive\\Desktop\\SCADA\\log files\\test LSTM_STD.txt'
val_base = 'C:\\Users\\michaelzaslavski\\OneDrive\\Desktop\\SCADA\\datasets\\validation'
KLSTM_test_log = 'C:\\Users\\michaelzaslavski\\OneDrive\\Desktop\\SCADA\\log files\\KLSTM detection test.txt'
LSTM_validation = val_base + '\\LSTM\\'
LSTM_based_OCSVM_log = logs + 'OCSVM train.txt'


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
        models_folders.append(binning_part + '_' + data_version)

    return models_folders, data_folders, binning_dict, params


def train_RF(RF_train_config_file_path, many=False):
    with open(RF_train_config_file_path, mode='r') as train_config:
        models_folders, data_folders, binning_dict, params = get_models_folders_data_folders(train_config)
        zipped = zip(models_folders, data_folders)
        for folder_pair in zipped:
            models_folder = folder_pair[0]
            data_folder = folder_pair[1]
            if not many:
                models.models.make_classifier(models_folder=models_folder, data_folder=data_folder,
                                              params=params, RF_only=True)
            else:
                for group in params['groups']:
                    models_folder = group + '_' + models_folder
                    data_folder = group + '_' + data_folder
                models.models.make_classifier(models_folder=models_folder, data_folder=data_folder,
                                              params=params, RF_only=True)


def train_OCSVM(OCSVM_train_config_file_path, group_info=None):
    with open(OCSVM_train_config_file_path, mode='r') as train_config:
        models_folders, data_folders, binning_dict, params = get_models_folders_data_folders(train_config)
        zipped = zip(models_folders, data_folders)
        for folder_pair in zipped:
            models_folder = folder_pair[0] if group_info is None else f'{group_info}_{folder_pair[0]}'
            data_folder = folder_pair[1] if group_info is None else f'{group_info}_{folder_pair[1]}'

            models.models.make_classifier(models_folder=models_folder, data_folder=data_folder,
                                          params=params, OCSVM_only=True, group_info=group_info)


def train_OCSVM_single_plc_split(train_config):
    for active_ip in data.active_ips:
        with open(LSTM_based_OCSVM_log, mode='a') as log:
            log.write(f'training OCSVM for PLC with IP: {active_ip}\n')
        group_info = f'single_plc_{active_ip}'
        train_OCSVM(train_config, group_info)


def train_OCSVM_all_plc_split(train_config):
    group_info = 'all_plcs'
    with open(LSTM_based_OCSVM_log, mode='a') as log:
        log.write('training OCSVM for all plcs split\n')
    train_OCSVM(train_config, group_info)


def train_OCSVM_pearson_split(train_config):
    for group in pearson_groups:
        with open(LSTM_based_OCSVM_log, mode='a') as log:
            log.write(f'training OCSVM for the pearson PLCs split with group: {group}\n')
        train_OCSVM(train_config, group)


def train_OCSVM_spearman_split(train_config):
    for group in spearman_groups:
        with open(LSTM_based_OCSVM_log, mode='a') as log:
            log.write(f'training OCSVM for the spearman PLCs split with group: {group}\n')
        train_OCSVM(train_config, group)


def train_OCSVM_k_means_split(train_config):
    for group in k_means_groups:
        with open(LSTM_based_OCSVM_log, mode='a') as log:
            log.write(f'training OCSVM for the kmeans PLCs split with group: {group}\n')
        train_OCSVM(train_config, group)


# single plc or groups of plcs.
def train_automaton(group=None, train_data_path=None, train_data=None):
    """
    :param group: group name, if none then we do it for a single plc.
                  otherwise, it can be used to train a PLC for each "real" group or for each PLC separately
                  it depends on the partitioning of the PLCs into group.
    :return:
    """
    if group is not None or train_data_path is not None:
        with open(train_data_path, mode='rb') as train_f:
            pkts = pickle.load(train_f)
    elif train_data is not None:
        pkts = train_data
    else:
        pkts = data.load(data.datasets_path, "TRAIN")

    bins, binning_methods, names = get_DFA_params()

    processed = data.process(data=pkts, name='v3_2_abstract', bins=None, binning=None, binner_path=None, scale=False,
                             registers=None, fill=False)
    registers = processed.columns[2:]

    training = pd.DataFrame(columns=['group', 'binning', '# bins', 'time'])

    binner_path = binners_base + '//DFA'
    if not os.path.exists(binner_path):
        Path(binner_path).mkdir(parents=True, exist_ok=True)

    for b in bins:
        for binning_method in binning_methods.keys():
            train_data = processed.copy()
            for col_name in processed.columns:

                if col_name in registers:
                    binner = binning_methods[binning_method]
                    binner_full_path = 'DFA//{}_{}_{}'.format(group, names[binning_method], b)
                    train_data[col_name] = binner(train_data, col_name, b, binner_full_path)

            dfa_input = squeeze(train_data)
            with open(DFA_log, mode='a') as log:
                log.write('Creating DFA for {}\n'.format(group))

            start = time.time()
            automaton = models.automaton.make_automaton(registers, dfa_input)
            end = time.time()

            result = {'group': group, 'binning': binning_method, '# bins': b, 'time': end - start}
            training_df = pd.DataFrame.from_dict(data={'0': result}, orient='index', columns=training.columns)
            training = pd.concat([training, training_df], axis=0, ignore_index=True)

            with open(DFA_log, mode='a') as log:
                log.write('Done, time elapsed:{}, binning: {}, # bins: {}\n'.format(end - start, binning_method, b))

            with open('//sise//home//zaslavsm//SCADA//DFA datasets//{}_train_{}_{}'.format(group, binning_method, b),
                      mode='wb') as train_p:
                pickle.dump(dfa_input, train_p)

            with open(data.automaton_path + '//{}_{}_{}'.format(group, names[binning_method], b),
                      mode='wb') as dfa_path:
                pickle.dump(automaton, dfa_path)
    with pd.ExcelWriter(xl_path, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
        h = True
        if 'DFA training times (many plcs)' in writer.sheets.keys():
            row = writer.sheets['DFA training times (many plcs)'].max_row
        if row > 0:
            h = False
        training.to_excel(writer, sheet_name='DFA training times (many plcs)', startrow=row, header=h)


def train_automatons_single_plcs_split():
    load_data = data.datasets_path + '//single_plc//'
    for active_ip in data.active_ips:
        train_automaton(group=f'single_plc_{active_ip}', train_data_path=load_data + f'{active_ip}_train')


def train_automatons_all_plcs_split():
    load_data = data.datasets_path + '//all_plcs//'
    train_automaton(group='all_plcs', train_data_path=load_data + 'test')


def train_automatons_pearson_split():
    load_data = data.datasets_path + '//pearson//'
    for group in spearman_groups:
        train_automaton(group=group, train_data_path=load_data + f'{group}_train')


def train_automatons_spearman_split():
    load_data = data.datasets_path + '//spearman//'
    for group in spearman_groups:
        train_automaton(group=group, train_data_path=load_data + f'{group}_train')


def train_automatons_k_means_split():
    load_data = data.datasets_path + '//k_means//'
    for group in k_means_groups:
        train_automaton(group=group, train_data_path=load_data + f'{group}_train')


def train_LSTM_transition_algo(FSTM_config, raw_data, group='', group_registers=None):
    """
    1. get data.
    2. process data using algorithm.
    3. train lstm on the output of step 2.
    """
    # load and unpack params.
    time_windows, min_supports, numbers_of_bins, binning_methods, series_lengths = get_FSTM_params(FSTM_config)

    for binning_method in binning_methods:
        for number_of_bins in numbers_of_bins:
            # load training data.
            with open(FSTM_data + '\\train\\X_{}_{}'.format(binning_method, number_of_bins), mode='wb') as f:
                train = pickle.load(f)
            for time_window in time_windows:
                # load algo results.
                times_p = FSTM_data + '\\complete_patterns\\train_times_{}_{}_{}'.format(binning_method,
                                                                                         number_of_bins,
                                                                                         time_window)

                indices_p = FSTM_data + '\\complete_patterns\\train_indices_{}_{}_{}'.format(binning_method,
                                                                                             number_of_bins,
                                                                                             time_window)

                with open(times_p, mode='rb') as p:
                    prev_times = pickle.load(p)

                with open(indices_p, mode='rb') as p:
                    prev_indices = pickle.load(p)

                for min_support in min_supports:  # load dict and extract features from sliding windows.
                    filtered_patterns_dict = FSTM_data + '\\filtered\\train_patterns_dict_{}_{}_{}_{}'
                    filtered_patterns_dict_p = filtered_patterns_dict.format(binning_method, number_of_bins,
                                                                             time_window,
                                                                             min_support)
                    with open(filtered_patterns_dict_p, mode='rb') as f:
                        p_dict = pickle.load(f)
                    for window in window_sizes:
                        LSTM_train = data.PLCDependeciesAlgorithm.extract_features_v3(train, p_dict, prev_indices,
                                                                                      prev_times, window)

                        for series_length in series_lengths:  # train LSTM models.
                            dump_model = data.modeles_path + '\\FSTM\\'
                            dump_df = data.datasets_path + '\\FSTM\\'
                            model_name = 'FSTM_{}_{}_{}_{}_{}'.format(binning_method, number_of_bins, time_window,
                                                                      min_support, series_length)
                            models.simple_LSTM(LSTM_train, series_length, 42, model_name, train=1.0,
                                               models_path=dump_model,
                                               data_path=dump_df)


def run_FSTM(FSTM_config):
    funcs_dict = {"k_means": data.k_means_binning, "equal_frequency": data.equal_frequency_discretization,
                  "equal_width": data.equal_width_discretization}

    # get params.
    time_windows, supports, numbers_of_bins, binning_methods, series_lengths = get_FSTM_params(FSTM_config)

    with open(data.datasets_path + '\\TRAIN', mode='rb') as p:  # load data.
        pkts = pickle.load(p)

    for binning_method in binning_methods:
        for number_of_bins in numbers_of_bins:
            binners_folder = binners_base + '\\{}_{}'.format(binning_method, number_of_bins)

            if not os.path.exists(binners_folder):
                Path(binners_folder).mkdir(parents=True, exist_ok=True)

            suffix = '{}_{}\\col'.format(binning_method, number_of_bins)

            train_data = data.process_data_v3(pkts, 8, funcs_dict[binning_method], number_of_bins, False)

            for col_name in train_data.columns:  # bin data.
                if 'time' not in col_name:
                    train_data[col_name] = funcs_dict[binning_method](train_data, col_name, number_of_bins, suffix)

            train_data = squeeze(train_data)  # squeeze states.

            with open(FSTM_data + '\\train\\X_{}_{}'.format(binning_method, number_of_bins), mode='wb') as f:
                pickle.dump(train_data, f)

            for time_window in time_windows:
                # call algo.
                # do it once with a very low support and filter the not frequent enough ones when testing.
                start = time.time()
                print(
                    'running fstm, bins {}, window {}, binning {}'.format(number_of_bins, time_window, binning_method))

                frequent_transitions, prev_times, prev_indices, longest, time_stamps = find_frequent_transitions_sequences(
                    train_data, time_window, 0.5)

                print('found (unfiltered) = {}\n'.format(len(frequent_transitions)))

                end = time.time()

                with open(FSTM_train_log, mode='a') as log:
                    log.write('FSTM time elapsed: {}\n'.format(end - start))

                patterns_folder = FSTM_data + '\\complete_patterns'

                if not os.path.exists(patterns_folder):
                    Path(patterns_folder).mkdir(exist_ok=True, parents=True)

                # save the results of the low support run.

                times_p = FSTM_data + '\\complete_patterns\\train_times_{}_{}_{}'.format(binning_method,
                                                                                         number_of_bins,
                                                                                         time_window)

                indices_p = FSTM_data + '\\complete_patterns\\train_indices_{}_{}_{}'.format(binning_method,
                                                                                             number_of_bins,
                                                                                             time_window)

                timestamps_p = FSTM_data + '\\complete_patterns\\train_timestamps_{}_{}_{}'.format(
                    binning_method, number_of_bins,
                    time_window)

                with open(
                        FSTM_data + '\\complete_patterns\\train_patterns_{}_{}_{}'.format(binning_method,
                                                                                          number_of_bins,
                                                                                          time_window),
                        mode='wb') as patterns_p:
                    pickle.dump(frequent_transitions, patterns_p)

                with open(times_p, mode='wb') as p:
                    pickle.dump(prev_times, p)

                with open(indices_p, mode='wb') as p:
                    pickle.dump(prev_indices, p)

                with open(timestamps_p, mode='wb') as p:
                    pickle.dump(time_stamps, p)

                # filter
                filtered_dir = FSTM_data + '\\filtered'

                if not os.path.exists(filtered_dir):
                    Path(filtered_dir).mkdir(exist_ok=True, parents=True)

                filtered_patterns = FSTM_data + '\\filtered\\train_patterns_{}_{}_{}_{}'
                filtered_patterns_dict = FSTM_data + '\\filtered\\train_patterns_dict_{}_{}_{}_{}'

                for min_sup in supports:  # filter the transitions.
                    freq_transitions = []
                    # save, go over all the patterns and sub-patterns.
                    for transition, times in prev_times.items():
                        s = (len(times) * 100) / len(train_data)
                        if s >= min_sup:
                            freq_transitions.append(transition)

                    print('support = {}, bins = {}, binning = {}, window = {}, found = {}\n'.format(min_sup,
                                                                                                    number_of_bins,
                                                                                                    binning_method,
                                                                                                    time_window,
                                                                                                    len(freq_transitions)))

                    filtered_patterns_p = filtered_patterns.format(binning_method, number_of_bins, time_window,
                                                                   min_sup)
                    filtered_patterns_dict_p = filtered_patterns_dict.format(binning_method, number_of_bins,
                                                                             time_window,
                                                                             min_sup)

                    # save the filtered patterns.
                    with open(filtered_patterns_p, mode='wb') as filtered_patterns_path:
                        pickle.dump(freq_transitions, filtered_patterns_path)

                    # create and save the filtered patterns' dictionary.
                    idx = 0
                    p_dict = dict()
                    for p in freq_transitions:
                        p_dict[idx] = p
                        idx += 1

                    with open(filtered_patterns_dict_p, mode='wb') as f:
                        pickle.dump(p_dict, f)


# IRRELEVANT FOR NOW. NEED TO DETERMINE DETECTION METHOD.
def train_RF_OCSVM_from_transition_algo_LSTM(classifier_config, group='', RF=True):
    with open(classifier_config, mode='r') as classifier_conf:
        params = yaml.load(classifier_conf, Loader=yaml.FullLoader)

    train_data_path = data.datasets_path + '\\{}_FSTM\\'.format(group)
    models_path = data.modeles_path + '\\{}_FSTM\\'.format(group)

    if RF:
        models.models.make_classifier(models_folder=models_path, data_folder=train_data_path, params=params,
                                      RF_only=True, OCSVM_only=False)
    else:
        models.models.make_classifier(models_folder=models_path, data_folder=train_data_path, params=params,
                                      RF_only=False, OCSVM_only=True)


# TRAIN DATA
def make_input_for_KL(TIRP_config_file_path):
    pkt_df = data.load(data.datasets_path, "TRAIN")

    # consider only response packets from the PLC.
    # plc_df = pkt_df.loc[(pkt_df['src_ip'] == IP)]
    with open(TIRP_config_file_path, mode='r') as train_config:
        params = yaml.load(train_config, Loader=yaml.FullLoader)

    discovery_params = params['TIRP_discovery_params']
    binning_methods = {'KMeans': TIRP.k_means_binning, 'EqualFreq': TIRP.equal_frequency_discretization,
                       'EqualWidth': TIRP.equal_width_discretization}
    numbers_of_bins = discovery_params['number_of_bins']

    for window in window_sizes:
        TIRP.make_input(pkt_df, binning_methods, numbers_of_bins, window, regs_to_use=data.most_used,
                        consider_last=True)


# VALIDATION DATA
def make_validation_or_test_input_for_KL(TIRP_config_file_path, dataset="VAL"):
    if dataset == "VAL":
        pkt_df = data.load(data.datasets_path, dataset)
    else:
        pkt_df = data.load(test_sets_base_folder, dataset)

    with open(TIRP_config_file_path, mode='r') as train_config:
        params = yaml.load(train_config, Loader=yaml.FullLoader)

    discovery_params = params['TIRP_discovery_params']
    binning_methods = {'KMeans': TIRP.k_means_binning, 'EqualFreq': TIRP.equal_frequency_discretization,
                       'EqualWidth': TIRP.equal_width_discretization}

    numbers_of_bins = discovery_params['number_of_bins']

    binning_methods_inv = {TIRP.k_means_binning: 'KMeans', TIRP.equal_frequency_discretization: 'EqualFreq',
                           TIRP.equal_width_discretization: 'EqualWidth'}

    binning = {TIRP.k_means_binning: 'kmeans', TIRP.equal_frequency_discretization: 'equal_frequency',
               TIRP.equal_width_discretization: 'equal_width'}

    for window in window_sizes:
        ready_symbols = {}
        ready_entities = {}

        for b in binning_methods.values():
            for k in numbers_of_bins:
                suffix = '//{}_{}_{}'.format(binning[b], k, window)
                path_sym = TIRP.KL_symbols + suffix

                with open(path_sym, mode='wb') as symbols_path:
                    symbols = pickle.load(symbols_path)

                path_ent = TIRP.KL_entities + suffix
                with open(path_ent, mode='wb') as entities_path:
                    entities = pickle.load(entities_path)

                ready_symbols[(binning_methods_inv[b], k)] = symbols
                ready_entities[(binning_methods_inv[b], k)] = entities

        if dataset is "VAL":
            data_path = val_base + '//KL//events'
        else:
            data_path = test_sets_base_folder + "//KLSTM//events"
        if not os.path.exists(data_path):
            Path(data_path).mkdir(exist_ok=True, parents=True)
        TIRP.make_input(pkt_df, binning_methods, numbers_of_bins, window, regs_to_use=data.most_used,
                        consider_last=True, ready_symbols=ready_symbols, ready_entities=ready_entities,
                        test_path=data_path)


def create_benign_test_input_files_for_KLSTM(input_creation_config):
    """

    :param input_creation_config: configuration file for creating test files.
    :return:None
    """
    make_validation_or_test_input_for_KL(input_creation_config, "TEST")


def create_validation_input_files_for_KLSTM(input_creation_config):
    """

    :param input_creation_config: configuration file for creating test files.
    :return:None
    """
    make_validation_or_test_input_for_KL(input_creation_config, "VAL")


# TEST DATA
def create_test_input_TIRP_files_for_KL(injection_config, input_creation_config):
    """
    make test data sets for KL.
    for each input creation option: create TIRP with all possible injection options.
    """
    binning_methods = {
        'KMeans': TIRP.k_means_binning}  # , 'EqualFreq': TIRP.equal_frequency_discretization, 'EqualWidth': TIRP.equal_width_discretization}

    binning_methods_inv = {TIRP.k_means_binning: 'kmeans', TIRP.equal_frequency_discretization: 'equal_frequency',
                           TIRP.equal_width_discretization: 'equal_width'}

    # first, grid over injection params.
    injection_lengths, step_overs, percentages, epsilons = get_injection_params(injection_config)

    with open(input_creation_config, mode='r') as TIRP_creation_config:
        TIRP_params = yaml.load(TIRP_creation_config, Loader=yaml.FullLoader)

    binning = TIRP_params['binning']
    bins = TIRP_params['number_of_bins']

    for injection_length in injection_lengths:
        for step_over in step_overs:
            anomaly_percentage = injection_length / (injection_length + step_over)
            if anomaly_percentage > lim:
                pass
            else:
                for percentage in percentages:
                    for epsilon in epsilons:
                        # load injected data.
                        df_path = test_sets_base_folder + '//raw//data_{}_{}_{}'.format(injection_length, step_over,
                                                                                        percentage)
                        labels_path = test_sets_base_folder + '//raw//labels_{}_{}_{}'.format(injection_length,
                                                                                              step_over,
                                                                                              percentage)
                        with open(df_path, mode='rb') as df_f:
                            anomalous_data = pickle.load(df_f)
                        with open(labels_path, mode='rb') as labels_f:
                            labels = pickle.load(labels_f)

                        test_path_sliding_windows = test_sets_base_folder + '//KL//test_events//{}_{}_{}'.format(
                            injection_length, step_over,
                            percentage)

                        for window_size in window_sizes:
                            ready_symbols = dict()
                            ready_entities = dict()

                            # create the labels for the detection.
                            # for each window: [start, end]
                            test_labels = []
                            for i in range(len(anomalous_data) - window_size + 1):
                                # get the labels for the windows' packets.
                                window_labels = labels[i: i + window_size]

                                # the label is 0 for a benign packet and 1 for an anomalous packets.
                                # so a set of packets has an anomaly in it iff the max of its corresponding labels is 1.
                                window_label = max(window_labels)

                                # add label.
                                test_labels.append(window_label)

                            for b in binning_methods.values():
                                for number_of_bins in bins:
                                    # discover events in separate windows.

                                    # make sure dirs exists.
                                    dir_path = test_sets_base_folder + '//KL//test_events'
                                    if not os.path.exists(dir_path):
                                        Path(dir_path).mkdir(parents=True, exist_ok=True)

                                    # discover TIRPs.
                                    # pass symbols and entities that were previously found.
                                    suffix = '//{}_{}_{}'.format(binning_methods_inv[b],
                                                                 number_of_bins, window_size)

                                    symbols_path = TIRP.KL_symbols + suffix
                                    entities_path = TIRP.KL_entities + suffix

                                    with open(symbols_path, mode='rb') as syms_path:
                                        symbols = pickle.load(syms_path)
                                    with open(entities_path, mode='rb') as ent_path:
                                        entities = pickle.load(ent_path)

                                    k = number_of_bins

                                    ready_symbols[(binning_methods_inv[b], k)] = symbols
                                    ready_entities[(binning_methods_inv[b], k)] = entities

                                    path = test_sets_base_folder + '//KL//test_labels//{}_{}_{}_{}_{}_{}'.format(
                                        binning_methods_inv[b],
                                        number_of_bins,
                                        window_size,
                                        injection_length,
                                        step_over,
                                        percentage)

                                    dir_path = test_sets_base_folder + '//KL//test_labels'
                                    if not os.path.exists(dir_path):
                                        Path(dir_path).mkdir(parents=True, exist_ok=True)

                                    with open(path, mode='wb') as labels_path:
                                        pickle.dump(test_labels, labels_path)
                            TIRP.make_input(anomalous_data, binning_methods, bins,
                                            window_size, consider_last=True, regs_to_use=data.most_used,
                                            test_path=test_path_sliding_windows,
                                            ready_symbols=ready_symbols, ready_entities=ready_entities)


def filter_TIRPs(KL_config_file_path):
    """
    we discover all the TIRPs with a very low support threshold and then filter out the ones having higher support
    to avoid running KarmaLego many times.
    :param KL_config_file_path: KL params
    :return:
    """
    # 1. get to the file containing the mined TIRPs with very low support.
    binning, bins, windows, epsilons, max_gaps, K_values, default_supp, kernels, nu = get_KL_params(KL_config_file_path)

    binning_times_bins = itertools.product(binning, bins)
    parent_folders = itertools.product(windows, binning_times_bins)

    for window_binning_bins in parent_folders:
        # parameters of the events making.
        window = window_binning_bins[0]
        binning = window_binning_bins[1][0]
        bins = window_binning_bins[1][1]

        windows_folders_folder = KL_base + "{}_bins_{}_window_{}_out".format(binning, bins, window)

        max_gaps_times_ks = itertools.product(max_gaps, K_values)
        KL_hyperparams = itertools.product(epsilons, max_gaps_times_ks)

        # parameters of KL.
        for eps_gap_supp in KL_hyperparams:
            epsilon = eps_gap_supp[0]
            max_gap = eps_gap_supp[1][0]
            k = eps_gap_supp[1][1]

            # we will save the filtered TIRPs here.
            dest_path_suffix = "//eps_{0}_k_{1}_maxGap_{2}".format(epsilon, k, max_gap)
            # path to read the whole set of TIRPs from.
            src_path_suffix = "//eps_{0}_minHS_{1}_maxGap_{2}".format(epsilon, default_supp, max_gap)

            windows_outputs_destination_folder_path = windows_folders_folder + dest_path_suffix
            windows_outputs_src_folder_path = windows_folders_folder + src_path_suffix

            if not os.path.exists(windows_outputs_destination_folder_path):
                Path(windows_outputs_destination_folder_path).mkdir(parents=True)

            # call helper function to finish.
            filter_and_write(windows_outputs_src_folder_path, windows_outputs_destination_folder_path, k)


# switch to filtering by a percentile!
def filter_and_write(windows_outputs_src_folder_path, windows_outputs_destination_folder_path, k):
    # iterate over the TIRPs in the windows files in the src folder.
    # read every TIRP line in the file and check for the HS. if it's high enough write it to the destination file.
    for TIRPs_in_window in os.listdir(windows_outputs_src_folder_path):
        window_file = windows_outputs_src_folder_path + '\\' + TIRPs_in_window
        dst_window_file = windows_outputs_destination_folder_path + '\\' + TIRPs_in_window
        numbers_of_instances = []
        for TIRP_line in window_file:
            tirp = TIRP.parse_line(TIRP_line)
            numbers_of_instances.append(TIRP.get_number_of_instances(tirp.instances))

        min_num_instances = sorted(numbers_of_instances)[k]

        for TIRP_line in window_file:
            # parse into TIRP object
            tirp = TIRP.parse_line(TIRP_line)
            # filter by horizontal support.
            if TIRP.get_number_of_instances(tirp.instances) >= min_num_instances:
                if not os.path.exists(dst_window_file):
                    Path(dst_window_file).mkdir(parents=True)
                # write TIRP.
                with open(dst_window_file, mode='a') as dst_p:
                    dst_p.write(TIRP_line + '\n')


def train_LSTMs_from_KL(KL_config_file_path):
    # go over all KL configurations:
    binning, bins, windows, epsilons, max_gaps, k_values, default_supp, kernels, nus = get_KL_params(
        KL_config_file_path)
    look_back = [20]

    for binning_method in binning:
        for number_of_bins in bins:
            for window_size in windows:

                windows_folders_folder = KL_base + "{}_{}_{}_out".format(binning_method, number_of_bins,
                                                                         window_size)

                for epsilon in epsilons:
                    for max_gap in max_gaps:
                        for k in k_values:

                            path_suffix = "//eps_{0}_k_{1}_maxGap_{2}".format(epsilon,
                                                                              k,
                                                                              max_gap)
                            windows_outputs_folder_path = windows_folders_folder + path_suffix
                            TIRP_path = TIRPs_base + '//{}_{}_{}_{}_{}_{}'.format(binning, bins, window_size, epsilon,
                                                                                  k,
                                                                                  max_gap)

                            # for each configuration: call parse_output.
                            TIRP_df = TIRP.output.parse_output(windows_outputs_folder_path, train=True,
                                                               tirps_path=TIRP_path)
                            # train LSTM.
                            for series_len in look_back:
                                model_name = '{}_{}_{}_{}_{}_{}'.format(binning_method, number_of_bins, window_size,
                                                                        epsilon, max_gap,
                                                                        k)
                                train_data_path = KL_base + 'KL_LSTM'
                                models_path = data.modeles_path + '//KL_LSTM'
                                with open(KL_LSTM_log, mode='a') as log:
                                    log.write('Training: {}_{}_{}_{}_{}_{}'.format(binning_method, number_of_bins,
                                                                                   window_size, epsilon, max_gap,
                                                                                   k))

                                start = time.time()
                                models.simple_LSTM(TIRP_df, series_len, 42, model_name, train=1,
                                                   models_path=models_path, data_path=train_data_path)
                                end = time.time()
                                with open(KL_LSTM_log, mode='a') as log:
                                    log.write("trained, time elapsed: {}".format(end - start))


def train_OCSVM_from_KL_LSTMs(KL_config_file_path):
    # go over all KL-LSTM configurations:
    # for each configuration: train RF, all the labels are 0.
    with open(KL_config_file_path, mode='r') as train_config:
        params = yaml.load(train_config, Loader=yaml.FullLoader)

    KL_params = params['KarmaLegoParams']
    binning = KL_params['BinningMethods']
    bins = KL_params['Bins']
    windows = KL_params['Windows']
    epsilons = KL_params['Epsilons']
    max_gaps = KL_params['MaxGaps']
    k_values = KL_params['Ks']
    look_back = [20]
    OCSVM_params = params['OCSVM']
    nus = OCSVM_params['nu']
    kernels = OCSVM_params['kernel']

    for binning_method in binning:
        for number_of_bins in bins:
            for window_size in windows:
                for epsilon in epsilons:
                    for max_gap in max_gaps:
                        for k in k_values:
                            # get LSTM and LSTM data.
                            for series_len in look_back:
                                model_name = '{}_{}_{}_{}_{}_{}_{}'.format(binning_method, number_of_bins, window_size,
                                                                           epsilon, max_gap, k,
                                                                           series_len)
                                train_data_folder_path = KL_base
                                models_path = data.modeles_path + '//KL_LSTM'
                                LSTM = keras.models.load_model(models_path + '//' + model_name)

                                with open(train_data_folder_path + '//X_train_{}'.format(model_name),
                                          mode='rb') as data_p:
                                    X_train = pickle.load(data_p)

                                predictions = LSTM.predict(X_train)
                                predictions_path = KL_OCSVM_datasets + '//X_train_{}_{}_{}_{}_{}_{}_{}'.format(
                                    binning_method, number_of_bins, window_size, epsilon, max_gap, k,
                                    series_len)
                                with open(predictions_path, mode='wb') as pred_f:
                                    pickle.dump(predictions, pred_f)

                                # all packets are considered benign. everything is labeled 0.
                                # train OCSVM to label predictions.
                                for kernel in kernels:
                                    for nu in nus:
                                        with open(KL_based_OCSVM_log, mode='a') as log:
                                            log.write('training:\n{}_{}_{}_{}_{}_{}_{}_{}_{}\n'.format(binning_method,
                                                                                                       number_of_bins,
                                                                                                       window_size,
                                                                                                       epsilon, max_gap,
                                                                                                       k,
                                                                                                       series_len,
                                                                                                       kernel, nu))
                                        ocsvm = OneClassSVM(kernel=kernel, nu=nu)
                                        start = time.time()
                                        ocsvm.fit(abs(predictions - X_train))
                                        end = time.time()
                                        with open(KL_based_OCSVM_log, mode='a') as log:
                                            log.write('Trained, time elapsed: {}'.format(end - start))
                                        # save model.
                                        model_path = KL_OCSVM_base + '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(binning_method,
                                                                                                         number_of_bins,
                                                                                                         window_size,
                                                                                                         epsilon,
                                                                                                         max_gap,
                                                                                                         k,
                                                                                                         series_len,
                                                                                                         kernel, nu)
                                        keras.models.save_model(ocsvm, model_path)


def create_train_sets_HTM(train_config, group_info=None):
    folders = {"k_means": 'KMeans', "equal_frequency": 'EqualFreq',
               "equal_width": 'EqualWidth'}
    # lstm params.
    methods, bins, data_versions = get_LSTM_params(train_config)

    for data_version in data_versions:
        folder_name = data_version['name']
        for number_of_bins in bins:
            for method in methods:
                method_folder = folders[method]
                version_desc = data_version['desc']
                data_folder_path = data.datasets_path + '//{}_{}'.format(method_folder, folder_name)
                if group_info is not None:
                    data_folder_path = data.datasets_path + '//{}_{}_{}'.format(group_info, method_folder, folder_name)
                dataset_name = 'X_train_{}_{}_{}'.format(version_desc, method, number_of_bins)
                with open(data_folder_path + '//' + dataset_name, mode='rb') as data_f:
                    train_df = pickle.load(data_f)
                y_train_path = 'y_train_{}_{}_{}'.format(version_desc, method, number_of_bins)
                with open(data_folder_path + '//' + y_train_path, mode='rb') as y_f:
                    y_train = pickle.load(y_f)
                train_df = np.concatenate([train_df[0], y_train])
                htm_df_base_path = HTM_base + '//datasets//' + '{}_{}'.format(method_folder, folder_name)
                if group_info is not None:
                    htm_df_base_path = HTM_base + '//datasets//' + '{}_{}_{}'.format(group_info, method_folder,
                                                                                     folder_name)
                htm_df_name = dataset_name
                if not os.path.exists(htm_df_base_path):
                    Path(htm_df_base_path).mkdir(exist_ok=True, parents=True)
                cols = data.process(None, folder_name, None, None, scale=True, binner_path=None,
                                    registers=data.most_used, fill=True, get_cols=True)
                write_df_to_csv(train_df, cols, htm_df_base_path + '//' + htm_df_name + '.csv')


def create_HTM_train_sets_single_plc_split(train_config):
    for ip in data.active_ips:
        group_info = f'single_plc_{ip}'
        create_train_sets_HTM(train_config, group_info=group_info)


def create_HTM_train_sets_all_plcs_split(train_config):
    create_train_sets_HTM(train_config, 'all_plcs')


def create_HTM_train_sets_spearman_split(train_config):
    for group in spearman_groups:
        create_train_sets_HTM(train_config, group_info=group)


def create_HTM_train_sets_pearson_split(train_config):
    for group in pearson_groups:
        create_train_sets_HTM(train_config, group_info=group)


def create_HTM_train_sets_k_means_split(train_config):
    for group in k_means_groups:
        create_train_sets_HTM(train_config, group_info=group)


def write_df_to_csv(df_arr, cols, path):
    with open(path, 'w', newline='') as train_file:
        train_writer = csv.writer(train_file)
        # write the field names.
        train_cols = cols
        train_writer.writerow(train_cols)
        # write the field types.
        train_cols_types = ['float'] * len(train_cols)
        train_writer.writerow(train_cols_types)
        # use no flags.
        train_writer.writerow([None] * len(train_cols))
    pd.DataFrame(df_arr).to_csv(path_or_buf=path, index=False, header=False, mode='a')


def create_val_sets_htm(train_config, group_info=None):
    folders = {"k_means": 'KMeans', "equal_frequency": 'EqualFreq',
               "equal_width": 'EqualWidth'}
    # lstm params.
    methods, bins, data_versions = get_LSTM_params(train_config)

    for data_version in data_versions:
        folder_name = data_version['name']
        for number_of_bins in bins:
            for method in methods:
                method_folder = folders[method]
                version_desc = data_version['desc']
                data_folder_path = val_base + '//LSTM//{}_{}'.format(folder_name, method_folder)
                if group_info is not None:
                    data_folder_path = val_base + '//LSTM//{}_{}_{}'.format(group_info, folder_name, method_folder)
                dataset_name = 'X_{}_{}_{}'.format(version_desc, method, number_of_bins)
                with open(data_folder_path + '//' + dataset_name, mode='rb') as data_f:
                    val_df = pickle.load(data_f)
                y_val_path = 'y_{}_{}_{}'.format(version_desc, method, number_of_bins)
                with open(data_folder_path + '//' + y_val_path, mode='rb') as y_f:
                    y_val = pickle.load(y_f)
                val_df = np.concatenate([val_df[0], y_val])
                htm_df_base_path = val_base + '//HTM//' + '{}_{}'.format(method_folder, folder_name)
                if group_info is not None:
                    htm_df_base_path = val_base + '//HTM//' + '{}_{}_{}'.format(group_info, method_folder, folder_name)
                htm_df_name = dataset_name
                if not os.path.exists(htm_df_base_path):
                    Path(htm_df_base_path).mkdir(exist_ok=True, parents=True)
                cols = data.process(None, folder_name, None, None, scale=True, binner_path=None,
                                    registers=data.most_used, fill=True, get_cols=True)
                write_df_to_csv(val_df, cols, htm_df_base_path + '//' + htm_df_name + '.csv')


def create_val_sets_htm_single_plc_split(train_config):
    for active_ip in data.active_ips:
        group_info = f'singe_plc_{active_ip}'
        create_val_sets_htm(train_config, group_info=group_info)


def create_val_sets_htm_all_plcs_split(train_config):
    create_val_sets_htm(train_config, group_info='all_plcs')


def create_val_sets_htm_spearman_split(train_config):
    for group in spearman_groups:
        create_val_sets_htm(train_config, group_info=group)


def create_val_sets_htm_pearson_split(train_config):
    for group in spearman_groups:
        create_val_sets_htm(train_config, group_info=group)


def create_val_sets_htm_k_means_split(train_config):
    for group in spearman_groups:
        create_val_sets_htm(train_config, group_info=group)


"""
The following function create the test files for the various classifiers.
Use configuration file to create them.
"""


def create_test_sets_for_HTM(injection_config, train_config, group_info=None, df_for_regs=None,
                             effected_plcs_suffix=None):
    injection_lengths, step_overs, percentages, epsilons = get_injection_params(injection_config)
    methods, bins, data_versions = get_LSTM_params(train_config)
    HTM_test_sets_base = test_sets_base_folder + '//HTM'

    for injection_length in injection_lengths:
        for step_over in step_overs:
            for percentage in percentages:
                for epsilon in epsilons:
                    for method in methods:
                        for number_of_bins in bins:
                            for data_version in data_versions:

                                test_sets_folder = test_sets_base_folder + '//LSTM//{}_{}_{}_{}//'.format(
                                    group_info, data_version['name'], method, number_of_bins)

                                p_suffix = '{}_{}_{}_{}_{}_{}'.format(
                                    data_version['desc'], method, number_of_bins, injection_length,
                                    step_over, percentage)

                                if effected_plcs_suffix is not None:
                                    p_suffix += f'_{effected_plcs_suffix}'

                                X_test_path = test_sets_folder + 'X_test_' + p_suffix
                                y_test_path = test_sets_folder + 'y_test_' + p_suffix
                                labels_path = test_sets_folder + 'labels_' + p_suffix

                                with open(X_test_path, mode='rb') as X_f:
                                    X_test = pickle.load(X_f)
                                with open(y_test_path, mode='rb') as y_f:
                                    y_test = pickle.load(y_f)
                                with open(labels_path, mode='rb') as labels_f:
                                    test_labels = pickle.load(labels_f)

                                test_df = np.concatenate([X_test[0], y_test])
                                htm_test_sets_folder = HTM_test_sets_base + '//{}_{}_{}//'.format(
                                    data_version['name'], method, number_of_bins)

                                if group_info is not None:
                                    htm_test_sets_folder = HTM_test_sets_base + '//{}_{}_{}_{}//'.format(
                                        group_info, data_version['name'], method, number_of_bins)

                                if not os.path.exists(htm_test_sets_folder):
                                    Path(htm_test_sets_folder).mkdir(exist_ok=True, parents=True)

                                HTM_X_test_path = htm_test_sets_folder + 'X_test_' + p_suffix
                                HTM_labels_path = htm_test_sets_folder + 'labels_' + p_suffix

                                cols = data.process(None, data_version['name'], None, None, scale=True,
                                                    binner_path=None, registers=data.most_used, fill=True,
                                                    get_cols=True, df_for_regs=df_for_regs)
                                write_df_to_csv(test_df, cols, HTM_X_test_path + '.csv')

                                with open(HTM_labels_path, mode='wb') as labels_f:
                                    pickle.dump(test_labels, labels_f, protocol=2)


def create_test_sets_for_HTM_single_plc_split(injection_config, train_config, effected_plcs_suffix=None):
    for active_ip in data.active_ips:
        df_for_regs_path = data.datasets_path + f'//single_plc//{active_ip}_train'
        with open(df_for_regs_path, mode='rb') as df_f:
            df_for_regs = pickle.load(df_f)
        create_test_sets_for_HTM(injection_config, train_config, f'single_plc_{active_ip}', df_for_regs=df_for_regs,
                                 effected_plcs_suffix=effected_plcs_suffix)


def create_test_sets_for_HTM_all_plcs_split(injection_config, train_config, effected_plcs_suffix=None):
    df_for_regs_path = data.datasets_path + f'//all_plcs//train'
    with open(df_for_regs_path, mode='rb') as df_f:
        df_for_regs = pickle.load(df_f)
    create_test_sets_for_HTM(injection_config, train_config, 'all_plcs', df_for_regs=df_for_regs,
                             effected_plcs_suffix=effected_plcs_suffix)


def create_test_sets_for_HTM_pearson_split(injection_config, train_config, effected_plcs_suffix=None):
    for group in pearson_groups:
        df_for_regs_path = data.datasets_path + f'//pearson//{group}_train'
        with open(df_for_regs_path, mode='rb') as df_f:
            df_for_regs = pickle.load(df_f)
        create_test_sets_for_HTM(injection_config, train_config, group, df_for_regs=df_for_regs,
                                 effected_plcs_suffix=effected_plcs_suffix)


def create_test_sets_for_HTM_spearman_split(injection_config, train_config, effected_plcs_suffix=None):
    for group in spearman_groups:
        df_for_regs_path = data.datasets_path + f'//spearman//{group}_train'
        with open(df_for_regs_path, mode='rb') as df_f:
            df_for_regs = pickle.load(df_f)
        create_test_sets_for_HTM(injection_config, train_config, group, df_for_regs=df_for_regs,
                                 effected_plcs_suffix=effected_plcs_suffix)


def create_test_sets_for_HTM_k_means_split(injection_config, train_config, effected_plcs_suffix=None):
    for group in k_means_groups:
        df_for_regs_path = data.datasets_path + f'//k_means//{group}_train'
        with open(df_for_regs_path, mode='rb') as df_f:
            df_for_regs = pickle.load(df_f)
        create_test_sets_for_HTM(injection_config, train_config, group, df_for_regs=df_for_regs,
                                 effected_plcs_suffix=effected_plcs_suffix)


def create_all_HTM_test_sets_with_injections_by_splits(injection_config, train_config):
    for effected_plcs, effected_plcs_suffix in effected_plcs_data_tuples:
        create_test_sets_for_HTM_single_plc_split(injection_config=injection_config, train_config=train_config,
                                                  effected_plcs_suffix=effected_plcs_suffix)
        create_test_sets_for_HTM_all_plcs_split(injection_config=injection_config, train_config=train_config,
                                                effected_plcs_suffix=effected_plcs_suffix)
        create_test_sets_for_HTM_pearson_split(injection_config=injection_config, train_config=train_config,
                                               effected_plcs_suffix=effected_plcs_suffix)
        create_test_sets_for_HTM_spearman_split(injection_config=injection_config, train_config=train_config,
                                                effected_plcs_suffix=effected_plcs_suffix)
        create_test_sets_for_HTM_k_means_split(injection_config=injection_config, train_config=train_config,
                                               effected_plcs_suffix=effected_plcs_suffix)


def create_test_file_for_FSTM(FSTM_config, raw_test_data_df, injection_config, group='', group_regs=None):
    # injection params.
    injection_lengths, step_overs, percentages, epsilons = get_injection_params(injection_config)

    funcs_dict = {"k_means": data.k_means_binning, "equal_frequency": data.equal_frequency_discretization,
                  "equal_width": data.equal_width_discretization}
    # fstm params.
    time_windows, min_supports, numbers_of_bins, binning_methods, series_lengths = get_FSTM_params(FSTM_config)

    for injection_length in injection_lengths:
        for step_over in step_overs:
            anomaly_percentage = injection_length / (injection_length + step_over)
            if anomaly_percentage > lim:
                pass
            else:
                for percentage in percentages:
                    for epsilon in epsilons:
                        # load raw test sets and labels.
                        parent = test_sets_base_folder + '\\raw'
                        data_path = parent + '\\data_{}_{}_{}_{}'.format(injection_length, step_over, percentage,
                                                                         epsilon)
                        labels_path = parent + '\\labels_{}_{}_{}_{}'.format(injection_length, step_over, percentage,
                                                                             epsilon)

                        with open(data_path, mode='rb') as d_p:
                            anomalous_data = pickle.load(d_p)

                        with open(labels_path, mode='rb') as l_p:
                            labels = pickle.load(l_p)

                        for binning_method in binning_methods:
                            for number_of_bins in numbers_of_bins:
                                processed = data.process_data_v3(anomalous_data, 8, funcs_dict[binning_method],
                                                                 number_of_bins, False)
                                # we need to know which transitions include anomalies to create the test labels.
                                # iterate over anomalous_data, if a packet is anomalous, mark the transition from its' corresponding
                                # state as anomalous.

                                FSTM_in = squeeze(processed)

                                transitions_labels, pkts_to_states = get_transitions_labels(anomalous_data, labels,
                                                                                            FSTM_in)

                                p_pkt_dict = test_sets_base_folder + '\\labels_{}_{}_{}_{}_{}'.format(
                                    binning_method, number_of_bins, injection_length,
                                    step_over,
                                    percentage)

                                with open(p_pkt_dict, mode='wb') as f:
                                    pickle.dump(pkts_to_states, f)

                                for time_window in time_windows:
                                    times_p = FSTM_data + '\\complete_patterns\\train_times_{}_{}_{}'.format(
                                        binning_method,
                                        number_of_bins,
                                        time_window)

                                    indices_p = FSTM_data + '\\complete_patterns\\train_indices_{}_{}_{}'.format(
                                        binning_method,
                                        number_of_bins,
                                        time_window)

                                    with open(times_p, mode='rb') as p:
                                        prev_times = pickle.load(p)

                                    with open(indices_p, mode='rb') as p:
                                        prev_indices = pickle.load(p)

                                    for min_support in min_supports:
                                        filtered_patterns_dict = FSTM_data + '\\filtered\\train_patterns_dict_{}_{}_{}_{}'
                                        filtered_patterns_dict_p = filtered_patterns_dict.format(binning_method,
                                                                                                 number_of_bins,
                                                                                                 time_window,
                                                                                                 min_support)
                                        with open(filtered_patterns_dict_p, mode='rb') as f:
                                            p_dict = pickle.load(f)
                                        for window in window_sizes:
                                            test_data = data.PLCDependeciesAlgorithm.extract_features_v3(FSTM_in,
                                                                                                         p_dict,
                                                                                                         prev_indices,
                                                                                                         prev_times,
                                                                                                         window)
                                            for series_length in series_lengths:
                                                X_test, y_test = models.custom_train_test_split(test_data,
                                                                                                series_length, 42, 1)
                                                suffix = '{}_{}_{}_{}_{}'.format(binning_method, number_of_bins,
                                                                                 time_window, min_support, window)
                                                X_test_p = test_sets_base_folder + '\\FSTM_X_test_' + suffix
                                                Y_test_p = test_sets_base_folder + '\\FSTM_Y_test_' + suffix
                                                # save x test and y test.
                                                with open(X_test_p, mode='wb') as f:
                                                    pickle.dump(X_test, f)
                                                with open(Y_test_p, mode='wb') as f:
                                                    pickle.dump(y_test, f)


def create_test_files_DFA(injection_config, group=None, df_for_regs=None, effected_plcs_suffix=None):
    """
     just grid over all the injection parameters and binning params.
    """

    binners = [data.equal_width_discretization]
    binner_to_str = {data.equal_width_discretization: 'equal_width'}
    folders_names = {'k_means': 'KMeans', 'equal_frequency': 'EqualFreq', 'equal_width': 'EqualWidth'}
    n_bins, binning_methods, names = get_DFA_params()

    with open(injection_config, mode='r') as anomalies_config:
        injection_params = yaml.load(anomalies_config, Loader=yaml.FullLoader)

    injection_lengths = injection_params['InjectionLength']
    step_overs = injection_params['StepOver']
    percentages = injection_params['Percentage']
    epsilons = injection_params['Epsilon']
    lim = 0.2

    for injection_length in injection_lengths:
        for step_over in step_overs:
            anomaly_percentage = injection_length / (injection_length + step_over)
            if anomaly_percentage > lim:
                pass
            else:
                for percentage in percentages:
                    for epsilon in epsilons:

                        parent = test_sets_base_folder + '//raw'
                        data_path = parent + '//{}_data_{}_{}_{}'.format(group, injection_length, step_over,
                                                                         percentage)
                        labels_path = parent + '//{}_labels_{}_{}_{}'.format(group, injection_length, step_over,
                                                                             percentage)

                        if effected_plcs_suffix is not None:
                            data_path += f'_{effected_plcs_suffix}'
                            labels_path += f'_{effected_plcs_suffix}'

                        with open(data_path, mode='rb') as d_p:
                            anomalous_data = pickle.load(d_p)
                        with open(labels_path, mode='rb') as l_p:
                            labels = pickle.load(l_p)

                        test_df = data.process(anomalous_data, 'v3_2_abstract', None, None, False, binner_path=None,
                                               registers=None, fill=False, df_for_regs=df_for_regs)
                        for binner in binners:
                            for bins in n_bins:

                                binned_test_df = test_df.copy()
                                for col_name in test_df.columns:
                                    if 'time' not in col_name:
                                        # load binner.
                                        binner_path = binners_base + '//DFA//{}_{}_{}_{}'.format(
                                            group, folders_names[binner_to_str[binner]], bins, col_name)
                                        with open(binner_path, mode='rb') as binner_p:
                                            col_binner = pickle.load(binner_p)
                                        binned_test_df[col_name] = binned_test_df[col_name].fillna(
                                            binned_test_df[col_name].mean())
                                        binned_test_df[col_name] = col_binner.transform(
                                            binned_test_df[col_name].to_numpy().reshape(-1, 1))

                                dfa_in = squeeze(binned_test_df)
                                # we need to know which transitions include anomalies to create the test labels.
                                # iterate over anomalous_data, if a packet is anomalous, mark the transition from its' corresponding
                                # state as anomalous.

                                transitions_labels, pkts_to_states = get_transitions_labels(anomalous_data, labels,
                                                                                            dfa_in)

                                p_x_test = test_sets_base_folder + '//DFA//{}_X_test_{}_{}_{}_{}_{}'.format(
                                    group, binner_to_str[binner], bins, injection_length,
                                    step_over,
                                    percentage)

                                p_labels = test_sets_base_folder + '//DFA//{}_labels_{}_{}_{}_{}_{}'.format(
                                    group, binner_to_str[binner], bins, injection_length,
                                    step_over,
                                    percentage)

                                p_pkt_dict = test_sets_base_folder + '//DFA//{}_dict_{}_{}_{}_{}_{}'.format(
                                    group, binner_to_str[binner], bins, injection_length,
                                    step_over,
                                    percentage)

                                if effected_plcs_suffix is not None:
                                    p_x_test += f'_{effected_plcs_suffix}'
                                    p_labels += f'_{effected_plcs_suffix}'
                                    p_pkt_dict += f'_{effected_plcs_suffix}'

                                if not os.path.exists(test_sets_base_folder + '//DFA'):
                                    Path(test_sets_base_folder + '//DFA').mkdir(parents=True, exist_ok=True)

                                with open(p_x_test, mode='wb') as test_path:
                                    pickle.dump(dfa_in, test_path)
                                with open(p_labels, mode='wb') as p_labels:
                                    pickle.dump(transitions_labels, p_labels)
                                with open(p_pkt_dict, mode='wb') as p_pkts_dict:
                                    pickle.dump(pkts_to_states, p_pkts_dict)


def create_DFA_test_sets_single_plc_split(injection_config, effected_plcs_suffix=None):
    for ip in data.active_ips:
        df_for_regs_path = data.datasets_path + f'//single_plc//{ip}_train'
        with open(df_for_regs_path, mode='rb') as df_f:
            df_for_regs = pickle.load(df_f)
        create_test_files_DFA(injection_config, group=f'single_plc_{ip}', df_for_regs=df_for_regs,
                              effected_plcs_suffix=effected_plcs_suffix)


def create_DFA_test_sets_all_plcs_split(injection_config, effected_plcs_suffix=None):
    df_for_regs_path = data.datasets_path + f'//all_plcs//train'
    with open(df_for_regs_path, mode='rb') as df_f:
        df_for_regs = pickle.load(df_f)
    create_test_files_DFA(injection_config, group='all_plcs', df_for_regs=df_for_regs,
                          effected_plcs_suffix=effected_plcs_suffix)


def create_DFA_test_sets_spearman_split(injection_config, effected_plcs_suffix=None):
    for group in spearman_groups:
        df_for_regs_path = data.datasets_path + f'//spearman//{group}_train'
        with open(df_for_regs_path, mode='rb') as df_f:
            df_for_regs = pickle.load(df_f)
        create_test_files_DFA(injection_config, group=group, df_for_regs=df_for_regs,
                              effected_plcs_suffix=effected_plcs_suffix)


def create_DFA_test_sets_pearson_split(injection_config, effected_plcs_suffix=None):
    for group in pearson_groups:
        df_for_regs_path = data.datasets_path + f'//pearson//{group}_train'
        with open(df_for_regs_path, mode='rb') as df_f:
            df_for_regs = pickle.load(df_f)

        create_test_files_DFA(injection_config, group=group, df_for_regs=df_for_regs,
                              effected_plcs_suffix=effected_plcs_suffix)


def create_DFA_test_sets_k_means_split(injection_config, effected_plcs_suffix=None):
    for group in k_means_groups:
        df_for_regs_path = data.datasets_path + f'//k_means//{group}_train'
        with open(df_for_regs_path, mode='rb') as df_f:
            df_for_regs = pickle.load(df_f)
        create_test_files_DFA(injection_config, group=group, df_for_regs=df_for_regs,
                              effected_plcs_suffix=effected_plcs_suffix)


def create_all_test_files_for_DFA_with_injections_by_splits(injection_config):
    for effected_plcs, effected_plcs_suffix in effected_plcs_data_tuples:
        create_DFA_test_sets_single_plc_split(injection_config, effected_plcs_suffix=effected_plcs_suffix)
        create_DFA_test_sets_all_plcs_split(injection_config, effected_plcs_suffix=effected_plcs_suffix)
        create_DFA_test_sets_pearson_split(injection_config, effected_plcs_suffix=effected_plcs_suffix)
        create_DFA_test_sets_spearman_split(injection_config, effected_plcs_suffix=effected_plcs_suffix)
        create_DFA_test_sets_k_means_split(injection_config, effected_plcs_suffix=effected_plcs_suffix)


# 1. after running KL on the test_events, filter TIRPs by support. REDUNDANT.
def filter_TIRPs_test_files(KL_config_file_path, injection_config):
    """
    we discover all the TIRPs with a very low support threshold and then filter out the ones having higher support
    to avoid running KarmaLego many times.
    :param injection_config: injections parameters.
    :param KL_config_file_path: KL params
    :return:
    """
    # 1. get to the file containing the mined TIRPs with very low support.
    # get params for kl and injections.
    binning, bins, windows, epsilons, max_gaps, k_values, default_supp, kernels, nus = get_KL_params(
        KL_config_file_path)

    injection_lengths, step_overs, percentages, injection_epsilons = get_injection_params(injection_config)

    binning_times_bins = itertools.product(binning, bins)
    parent_folders = itertools.product(windows, binning_times_bins)

    for injection_length in injection_lengths:
        for step_over in step_overs:
            rate = injection_length / (injection_length + step_over)
            if rate > lim:
                pass
            else:
                for percentage in percentages:
                    for injection_epsilon in injection_epsilons:
                        for window_binning_bins in parent_folders:
                            # parameters of the events making.
                            window = window_binning_bins[0]
                            binning = window_binning_bins[1][0]
                            bins = window_binning_bins[1][1]
                            # \\{0}_{1}_{2}_{3}_{4}_{5}_{6}", method, num_bins, windowSize, length, step, percentage);
                            windows_folders_folder = KL_test_sets_base + "//{}_{}_{}_{}_{}_{}".format(binning, bins,
                                                                                                      window,
                                                                                                      injection_length,
                                                                                                      step_over,
                                                                                                      percentage)

                            max_gaps_times_k_values = itertools.product(max_gaps, k_values)
                            KL_hyperparams = itertools.product(epsilons, max_gaps_times_k_values)

                            # parameters of KL.
                            for eps_gap_supp in KL_hyperparams:
                                epsilon = eps_gap_supp[0]
                                max_gap = eps_gap_supp[1][0]
                                k = eps_gap_supp[1][1]

                                # we will save the filtered TIRPs here.
                                dest_path_suffix = "//{}_{}_{}".format(epsilon, k, max_gap)
                                # path to read the whole set of TIRPs from.
                                src_path_suffix = "//{}_{}_{}".format(epsilon, default_supp, max_gap)

                                windows_outputs_destination_folder_path = windows_folders_folder + dest_path_suffix
                                windows_outputs_src_folder_path = windows_folders_folder + src_path_suffix

                                if not os.path.exists(windows_outputs_destination_folder_path):
                                    Path(windows_outputs_destination_folder_path).mkdir(parents=True)

                                # call helper function to finish.
                                filter_and_write(windows_outputs_src_folder_path,
                                                 windows_outputs_destination_folder_path, k)


# 2. after filtering, make dfs for the LSTM.
def create_test_sets_KLSTM(KL_config_path, injections_config_path):
    # 1. go over kl params
    binning_methods, bins, windows, epsilons, max_gaps, k_values, default_supp, kernels, nus = get_KL_params(
        KL_config_path)

    # 2. go over injection paras
    injection_lengths, step_overs, percentages, injection_epsilons = get_injection_params(injections_config_path)

    # used for file accessing
    binning_times_bins = itertools.product(binning_methods, bins)
    parent_folders = itertools.product(windows, binning_times_bins)

    # 3. find the tirps folder.
    for injection_length in injection_lengths:
        for step_over in step_overs:
            rate = injection_length / (injection_length + step_over)
            if rate > lim:
                pass
            else:
                for percentage in percentages:
                    for injection_epsilon in injection_epsilons:
                        for window_binning_bins in parent_folders:
                            # parameters of the events making.
                            window = window_binning_bins[0]
                            binning = window_binning_bins[1][0]
                            bins = window_binning_bins[1][1]

                            test_windows_folders_folder = KL_test_sets_base + "//{}_{}_{}_{}_{}_{}".format(binning,
                                                                                                           bins,
                                                                                                           window,
                                                                                                           injection_length,
                                                                                                           step_over,
                                                                                                           percentage)

                            max_gaps_times_k_values = itertools.product(max_gaps, k_values)
                            KL_hyperparams = itertools.product(epsilons, max_gaps_times_k_values)

                            # parameters of KL.
                            for eps_gap_supp in KL_hyperparams:
                                epsilon = eps_gap_supp[0]
                                max_gap = eps_gap_supp[1][0]
                                k = eps_gap_supp[1][1]

                                # we will save the filtered TIRPs here.
                                test_path_suffix = "//{}_{}_{}".format(epsilon, k, max_gap)

                                # folder of tirps text files.
                                test_windows_outputs_folder_path = test_windows_folders_folder + test_path_suffix

                                # need to get the file of the TIRPs in the matching train set. pass it as tirps path.

                                TIRP_path = TIRPs_base + '//{}_{}_{}_{}_{}_{}'.format(binning, bins, window,
                                                                                      epsilon,
                                                                                      default_supp,
                                                                                      max_gap)
                                # call parse outout and save.
                                test_df = TIRP.output.parse_output(test_windows_outputs_folder_path,
                                                                   tirps_path=TIRP_path, train=False)

                                x_test, y_test = models.custom_train_test_split(test_df, 20, 42, 1)

                                # save df.
                                test_df_path_dir = test_sets_base_folder + '//KL//KL_LSTM'

                                if not os.path.exists(test_df_path_dir):
                                    Path(test_df_path_dir).mkdir(parents=True, exist_ok=True)

                                suffix = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(binning,
                                                                             bins, window,
                                                                             k,
                                                                             epsilon,
                                                                             max_gap,
                                                                             injection_length,
                                                                             step_over,
                                                                             percentage)

                                X_test_df_path = test_df_path_dir + '//X_test_' + suffix
                                y_test_df_path = test_df_path_dir + '//y_test_' + suffix

                                with open(X_test_df_path, mode='wb') as test_f:
                                    pickle.dump(x_test, test_f)
                                with open(y_test_df_path, mode='wb') as y_test_path:
                                    pickle.dump(y_test, y_test_path)


def create_KLSTM_test_or_val_files(KL_config_path, mode="TEST"):
    # 1. go over kl params
    binning_methods, bins, windows, epsilons, max_gaps, k_values, default_supp, kernels, nus = get_KL_params(
        KL_config_path)

    """
    String windows_folder = input_base + "//" + method + String.Format("_{0}_{1}", number_of_bins, window_size);
                                    String separated_output_folder = windows_folder + "_out";
    String outFile = separated_output_folder + "//" + String.Format("eps_{0}_minHS_{1}_maxGap_{2}",
                                                        epsilon, minHSup, maxGap) + "//" + String.Format("window#_{0}.txt", window_number);
    String input_base = test_sets_base_folder + "//KLSTM//events"
    """
    if mode == "TEST":
        input_base = test_sets_base_folder + "//KLSTM//events"
        data_base = test_sets_base_folder + "//KLSTM"
    else:
        input_base = val_base + "//KL//events"
        data_base = val_base + "//KL//KLSTM"

    for binning_method in binning_methods:
        for number_of_bins in bins:
            for window_size in windows:
                for KL_epsilon in epsilons:
                    for max_gap in max_gaps:
                        windows_folder = input_base + "//" + binning_method + "_{}_{}".format(number_of_bins,
                                                                                              window_size)
                        separated_output_folder = windows_folder + "_out"

                        for k_val in k_values:
                            TIRP_path = TIRPs_base + '//{}_{}_{}_{}_{}_{}'.format(binning_method, bins, window_size,
                                                                                  KL_epsilon,
                                                                                  k_val,
                                                                                  max_gap)

                            # for each configuration: call parse_output.
                            TIRP_df = TIRP.output.parse_output(separated_output_folder, train=False,
                                                               tirps_path=TIRP_path)
                            X, y = models.custom_train_test_split(TIRP_df, 20, 42, 1)
                            suffix = "{}_{}_{}_{}_{}_{}".format(binning_method, bins,
                                                                window_size, KL_epsilon,
                                                                k_val, max_gap)
                            x_p = data_base + "//KLSTM//X_test_" + suffix
                            y_p = data_base + "//KLSTM//y_test_" + suffix

                            with open(x_p, mode='wb') as x_f:
                                pickle.dump(X, x_f)
                            with open(y_p, mode='wb') as y_f:
                                pickle.dump(y, y_f)


def create_KLSTM_val_files(KL_config):
    create_KLSTM_test_or_val_files(KL_config, "VAL")


def create_KLSTM_test_files(KL_config):
    create_KLSTM_test_or_val_files(KL_config, "TEST")


def test_KLSTM_prediction(KL_config_path):
    """

    :param KL_config_path: configuration file for the KLSTM parameters.
    :return:
    """

    # go over the parameters:
    # get model and data
    # predict
    # log
    # go over all KL configurations:
    binning, bins, windows, epsilons, max_gaps, k_values, default_supp, kernels, nus = get_KL_params(KL_config_path)
    look_back = [20]

    results = pd.DataFrame(columns=['binning', '# bins', 'window size', 'epsilon', 'max gap', 'k', 'mse', 'r2'])

    for binning_method in binning:
        for number_of_bins in bins:
            for window_size in windows:
                for KL_epsilon in epsilons:
                    for max_gap in max_gaps:
                        for k_val in k_values:
                            # train LSTM.
                            for series_len in look_back:
                                model_name = '{}_{}_{}_{}_{}_{}'.format(binning_method, number_of_bins, window_size,
                                                                        KL_epsilon, max_gap,
                                                                        k_val)
                                test_data_path = KL_base + 'KL_LSTM'
                                models_path = data.modeles_path + '//KL_LSTM'
                                LSTM = keras.models.load_model(models_path + "//" + model_name)
                                suffix = "{}_{}_{}_{}_{}_{}".format(binning_method, bins,
                                                                    window_size,
                                                                    KL_epsilon,
                                                                    k_val, max_gap)
                                x_test_p = test_sets_base_folder + "//KLSTM//X_train_" + suffix
                                y_test_p = test_sets_base_folder + "//KLSTM//y_train_" + suffix

                                with open(x_test_p, mode='rb') as x_f:
                                    X_test = pickle.load(x_f)
                                with open(y_test_p, mode='rb') as y_f:
                                    y_test = pickle.load(y_f)

                                prediction = LSTM.predict(X_test)
                                mse = mean_squared_error(y_test, prediction)
                                r2 = r2_score(y_test, prediction)
                                result = {'binning': binning_method,
                                          '# bins': number_of_bins,
                                          'window size': window_size,
                                          'epsilon': KL_epsilon,
                                          'max gap': max_gap,
                                          'k': k_val,
                                          'mse': mse, 'r2': r2}
                                result_df = pd.DataFrame.from_dict(data={'0': result}, orient='index',
                                                                   columns=results.columns)
                                results = pd.concat([results, result_df], axis=0, ignore_index=True)

    with pd.ExcelWriter(xl_path, mode="a", engine="openpyxl",
                        if_sheet_exists="overlay") as writer:
        results.to_excel(writer, sheet_name='KLSTM predictions results')


# 1. after running KL on the test_events, filter TIRPs by support.
def filter_TIRPs_validation_files(KL_config_file_path):
    """
    we discover all the TIRPs with a very low support threshold and then filter out the ones having higher support
    to avoid running KarmaLego many times.
    :param KL_config_file_path: KL params
    :return:
    """
    # 1. get to the file containing the mined TIRPs with very low support.
    # get params for kl and injections.
    binning, bins, windows, epsilons, max_gaps, k_values, default_supp, kernels, nus = get_KL_params(
        KL_config_file_path)

    binning_times_bins = itertools.product(binning, bins)
    parent_folders = itertools.product(windows, binning_times_bins)

    for window_binning_bins in parent_folders:
        # parameters of the events making.
        window = window_binning_bins[0]
        binning = window_binning_bins[1][0]
        bins = window_binning_bins[1][1]
        # \\{0}_{1}_{2}_{3}_{4}_{5}_{6}", method, num_bins, windowSize, length, step, percentage);
        windows_folders_folder = val_base + "//KL//events" + "//{}_{}_{}".format(binning, bins, window)

        max_gaps_times_k_values = itertools.product(max_gaps, k_values)
        KL_hyperparams = itertools.product(epsilons, max_gaps_times_k_values)

        # parameters of KL.
        for eps_gap_supp in KL_hyperparams:
            epsilon = eps_gap_supp[0]
            max_gap = eps_gap_supp[1][0]
            k = eps_gap_supp[1][1]

            # we will save the filtered TIRPs here.
            dest_path_suffix = "//{}_{}_{}".format(epsilon, k, max_gap)
            # path to read the whole set of TIRPs from.
            src_path_suffix = "//{}_{}_{}".format(epsilon, default_supp, max_gap)

            windows_outputs_destination_folder_path = windows_folders_folder + dest_path_suffix
            windows_outputs_src_folder_path = windows_folders_folder + src_path_suffix

            if not os.path.exists(windows_outputs_destination_folder_path):
                Path(windows_outputs_destination_folder_path).mkdir(parents=True)

            # call helper function to finish.
            filter_and_write(windows_outputs_src_folder_path,
                             windows_outputs_destination_folder_path, k)


def create_validation_sets_for_KLSTM(KL_config):
    binning, bins, windows, epsilons, max_gaps, k_values, default_supp, kernels, nus = get_KL_params(
        KL_config)
    """
    String windows_folder = input_base + "//" + method + String.Format("_{0}_{1}", number_of_bins, window_size);
                                    String separated_output_folder = windows_folder + "_out";
                                    String outFile = separated_output_folder + "//" + String.Format("eps_{0}_minHS_{1}_maxGap_{2}",
                                                                                        epsilon, minHSup, maxGap) + "//" + String.Format("window#_{0}.txt", window_number);
    """
    input_base = val_base + "//KL//events"
    for binning_method in binning:
        for number_of_bins in bins:
            for window_size in windows:
                for KL_epsilon in epsilons:
                    for max_gap in max_gaps:
                        windows_folder = input_base + "//" + binning_method + "_{}_{}".format(number_of_bins,
                                                                                              window_size)
                        separated_output_folder = windows_folder + "_out"

                        for k_val in k_values:
                            TIRP_path = TIRPs_base + '//{}_{}_{}_{}_{}_{}'.format(binning_method, bins, window_size,
                                                                                  KL_epsilon,
                                                                                  k_val,
                                                                                  max_gap)

                            # for each configuration: call parse_output.
                            TIRP_test_df = TIRP.output.parse_output(separated_output_folder, train=False,
                                                                    tirps_path=TIRP_path)
                            X_test, y_test = models.custom_train_test_split(TIRP_test_df, 20, 42, 1)
                            suffix = "{}_{}_{}_{}_{}_{}".format(binning_method, bins,
                                                                window_size, KL_epsilon,
                                                                k_val, max_gap)
                            x_test_p = test_sets_base_folder + "//KLSTM//X_test_" + suffix
                            y_test_p = test_sets_base_folder + "//KLSTM//y_test_" + suffix

                            with open(x_test_p, mode='wb') as x_f:
                                pickle.dump(X_test, x_f)
                            with open(y_test_p, mode='wb') as y_f:
                                pickle.dump(y_test, y_f)


def filter_TIRPs_benign_test_files(KL_config_file_path):
    """
    we discover all the TIRPs with a very low support threshold and then filter out the ones having higher support
    to avoid running KarmaLego many times.
    :param KL_config_file_path: KL params
    :return:
    """
    # 1. get to the file containing the mined TIRPs with very low support.
    # get params for kl and injections.
    binning, bins, windows, epsilons, max_gaps, k_values, default_supp, kernels, nus = get_KL_params(
        KL_config_file_path)

    binning_times_bins = itertools.product(binning, bins)
    parent_folders = itertools.product(windows, binning_times_bins)

    for window_binning_bins in parent_folders:
        # parameters of the events making.
        window = window_binning_bins[0]
        binning = window_binning_bins[1][0]
        bins = window_binning_bins[1][1]
        # \\{0}_{1}_{2}_{3}_{4}_{5}_{6}", method, num_bins, windowSize, length, step, percentage);
        windows_folders_folder = KL_test_sets_base + "//{}_{}_{}".format(binning, bins, window)

        max_gaps_times_k_values = itertools.product(max_gaps, k_values)
        KL_hyperparams = itertools.product(epsilons, max_gaps_times_k_values)

        # parameters of KL.
        for eps_gap_supp in KL_hyperparams:
            epsilon = eps_gap_supp[0]
            max_gap = eps_gap_supp[1][0]
            k = eps_gap_supp[1][1]

            # we will save the filtered TIRPs here.
            dest_path_suffix = "//{}_{}_{}".format(epsilon, k, max_gap)
            # path to read the whole set of TIRPs from.
            src_path_suffix = "//{}_{}_{}".format(epsilon, default_supp, max_gap)

            windows_outputs_destination_folder_path = windows_folders_folder + dest_path_suffix
            windows_outputs_src_folder_path = windows_folders_folder + src_path_suffix

            if not os.path.exists(windows_outputs_destination_folder_path):
                Path(windows_outputs_destination_folder_path).mkdir(parents=True)

            # call helper function to finish.
            filter_and_write(windows_outputs_src_folder_path,
                             windows_outputs_destination_folder_path, k)


def test_KLSTM_detection(KL_config_path, injection_config):
    binning, bins, windows, epsilons, max_gaps, k_values, default_supp, kernels, nus = get_KL_params(KL_config_path)
    injection_lengths, step_overs, percentages, injection_epsilons = get_injection_params(injection_config)
    look_back = [20]

    folders = {"k_means": 'KMeans', "equal_frequency": 'EqualFreq',
               "equal_width": 'EqualWidth'}

    results_df = pd.DataFrame(columns=excel_cols)
    labels_df = pd.DataFrame(columns=['binning', '# bins', 'window size', 'KL epsilon', 'minimal K', 'max gap',
                                      'injection length', 'step over', 'percentage', '# window', 'model label',
                                      'true label'])

    # 4. detect and set threshold on validation set.
    # 5. for each window, std count -> detect.
    for binning_method in binning:
        for number_of_bins in bins:
            for window_size in windows:
                for KL_epsilon in epsilons:
                    for max_gap in max_gaps:
                        for k_val in k_values:
                            # get LSTM and LSTM data.
                            for series_len in look_back:
                                with open(KLSTM_test_log, mode='a') as log:
                                    log.write('testing KLSTM with parameters:\n')
                                    log.write(
                                        'binning:{}, # bins:{}, window:{}, kl eps:{}, max gap:{}, k_val:{}\n'.format(
                                            binning_method, number_of_bins, window_size, KL_epsilon, max_gap, k_val))

                                model_name = '{}_{}_{}_{}_{}_{}_{}'.format(binning_method, number_of_bins, window_size,
                                                                           KL_epsilon, max_gap, k_val,
                                                                           series_len)
                                models_path = data.modeles_path + '//KL_LSTM'
                                LSTM = keras.models.load_model(models_path + '//' + model_name)

                                suffix = "{}_{}_{}_{}_{}_{}".format(binning_method, bins,
                                                                    window_size, KL_epsilon,
                                                                    k_val, max_gap)
                                data_base = val_base + "//KL//KLSTM"
                                x_val_p = data_base + "//KLSTM//X_test_" + suffix
                                y_val_p = data_base + "//KLSTM//y_test_" + suffix

                                with open(x_val_p, mode='wb') as x_f:
                                    X_val = pickle.load(x_f)
                                with open(y_val_p, mode='wb') as y_f:
                                    y_val = pickle.load(y_f)

                                LSTM_val_preds = LSTM.predict(X_val)

                                for kernel in kernels:
                                    for nu in nus:
                                        with open(KLSTM_test_log, mode='a') as log:
                                            log.write('testing with OCSVM with parameters:\n')
                                            log.write("kernel:{}, nu:{}\n".format(kernel, nu))
                                        # load model.
                                        model_path = KL_OCSVM_base + '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(binning_method,
                                                                                                         number_of_bins,
                                                                                                         window_size,
                                                                                                         KL_epsilon,
                                                                                                         max_gap,
                                                                                                         k_val,
                                                                                                         series_len,
                                                                                                         kernel, nu)
                                        OCSVM = keras.models.load_model(model_path)
                                        OCSVM_val_classification = OCSVM.predict(abs(LSTM_val_preds - y_val))

                                        for injection_length in injection_lengths:
                                            for step_over in step_overs:
                                                rate = injection_length / (injection_length + step_over)
                                                if rate > lim:
                                                    pass
                                                else:
                                                    for percentage in percentages:
                                                        for injection_epsilon in injection_epsilons:
                                                            suffix = "{}_{}_{}_{}_{}_{}".format(binning_method, bins,
                                                                                                window_size, KL_epsilon,
                                                                                                k_val, max_gap)
                                                            x_test_p = test_sets_base_folder + "//KLSTM//X_test_" + suffix
                                                            y_test_p = test_sets_base_folder + "//KLSTM//y_test_" + suffix

                                                            with open(x_test_p, mode='rb') as x_f:
                                                                X_test = pickle.load(x_f)
                                                            with open(y_test_p, mode='rb') as y_f:
                                                                y_test = pickle.load(y_f)

                                                            LSTM_predictions = LSTM.predict(X_test)
                                                            OCSVM_classification = OCSVM.predict(
                                                                abs(LSTM_predictions - y_test))

                                                            true_labels_path = test_sets_base_folder + '//KL//test_labels//{}_{}_{}_{}_{}_{}'.format(
                                                                binning_method,
                                                                number_of_bins,
                                                                window_size,
                                                                injection_length,
                                                                step_over,
                                                                percentage)

                                                            with open(true_labels_path, mode='r') as labels_f:
                                                                true_labels = pickle.load(labels_f)

                                                            mean, std = count_outliers(OCSVM_val_classification,
                                                                                       window_size)
                                                            for std_count in num_stds_count:
                                                                count_threshold = mean + std_count * std
                                                                true_windows_labels, model_windows_labels = LSTM_preds_to_window_preds(
                                                                    OCSVM_classification, true_labels, window_size,
                                                                    count_threshold)

                                                                # calculate metrics and log everything.
                                                                precision, recall, auc_score, f1, prc_auc_score, tn, fp, fn, tp = get_metrics(
                                                                    y_true=true_windows_labels,
                                                                    y_pred=model_windows_labels)

                                                                result = {'binning': binning_method,
                                                                          '# bins': bins,
                                                                          'KL epsilon': KL_epsilon,
                                                                          'max gap': max_gap,
                                                                          'minimal K': k_val,
                                                                          '# std count': std_count,
                                                                          'window size': window_size,
                                                                          'injection length': injection_length,
                                                                          'step over': step_over,
                                                                          'percentage': percentage,
                                                                          'precision': precision,
                                                                          'recall': recall,
                                                                          'auc': auc_score,
                                                                          'f1': f1,
                                                                          'prc': prc_auc_score}

                                                                """labels_test_df = pd.DataFrame(columns=labels_df.columns)
                                                                labels_test_df['# window'] = [i for i in range(
                                                                    len(true_windows_labels))]
                                                                labels_test_df['window size'] = window_size
                                                                labels_test_df['model label'] = model_windows_labels
                                                                labels_test_df['true label'] = true_windows_labels
                                                                labels_test_df['binning'] = folders[binning_method]
                                                                labels_test_df['# bins'] = bins
                                                                labels_test_df['# std count'] = std_count
                                                                labels_test_df['injection length'] = injection_length
                                                                labels_test_df['step over'] = step_over
                                                                labels_test_df['percentage'] = percentage
                                                                labels_test_df['KL epsilon'] = KL_epsilon
                                                                labels_test_df['minimal K'] = k_val
                                                                labels_test_df['max gap'] = max_gap

                                                                with pd.ExcelWriter(xl_path, mode="a",
                                                                                    engine="openpyxl",
                                                                                    if_sheet_exists="overlay") as writer:
                                                                    row = 0
                                                                    if 'KLSTM windows labels' in writer.sheets.keys():
                                                                        row = writer.sheets[
                                                                             'KLSTM windows labels'].max_row
                                                                    labels_test_df.to_excel(writer,
                                                                                            sheet_name='KLSTM windows labels',
                                                                                            startrow=row)"""

                                                                for col_name in excel_cols:
                                                                    if col_name not in DFA_cols:
                                                                        result[col_name] = '-'
                                                                results_df = pd.concat(
                                                                    [results_df,
                                                                     pd.DataFrame.from_dict(data={'0': result},
                                                                                            columns=excel_cols,
                                                                                            orient='index')],
                                                                    axis=0, ignore_index=True)
                                                                with open(KLSTM_test_log, mode='a') as log:
                                                                    log.write(
                                                                        'scores for injection with %:{}, len:{}, step over:{}\n'.format(
                                                                            percentage, injection_length, step_over))
                                                                    log.write(
                                                                        'auc:{}, f1:{}, prc:{}, precision:{}, recall:{},tn:{}, fp:{}, fn:{}, tp:{}\n'.format(
                                                                            auc_score, f1,
                                                                            prc_auc_score,
                                                                            precision,
                                                                            recall, tn, fp, fn, tp))
    with pd.ExcelWriter(xl_path, mode="a",
                        engine="openpyxl",
                        if_sheet_exists="overlay") as writer:
        results_df.to_excel(writer, sheet_name='KLSTM scores')


def make_best(results_df):
    # df for best scores without hyper-parameters being taken into consideration.
    best_df = pd.DataFrame(
        columns=['data version', 'binning', '# bins', 'precision', 'recall', 'auc', 'f1', 'prc',
                 'injection length',
                 'step over', 'percentage'])
    # group by the non-hyper-parameters and get the best results.
    grouped_results = results_df.groupby(by=['data version', 'binning', '# bins', 'injection length',
                                             'step over', 'percentage'])

    for group_name, group in grouped_results:
        best_precision = max(group['precision'])
        best_recall = max(group['recall'])
        best_auc = max(group['auc'])
        best_f1 = max(group['f1'])
        best_prc = max(group['prc'])

        best_result = {'data version': group_name[0], 'binning': group_name[1], '# bins': group_name[2],
                       'precision': best_precision, 'recall': best_recall, 'auc': best_auc, 'f1': best_f1,
                       'prc': best_prc,
                       'injection length': group_name[3],
                       'step over': group_name[4], 'percentage': group_name[5]}
        temp_df = pd.DataFrame.from_dict(data={'0': best_result}, orient='index', columns=best_df.columns)
        best_df = pd.concat([best_df, temp_df])
    return best_df


#################################IRRELEVANT FOR NOW#####################################################
def test_LSTM_based_classifiers(lstm_config, ocsvm_config, injection_config, group=''):
    """
    injection sets folder: folder_name, name, number_of_bins = binning method, data version, # bins
    path to RF = SCADA_BASE + '\\RFs\\' + 'diff_' + models_folder + '\\' + diff_' + model_name + 'estimators_{}_'.format(
                       estimators) + 'criterion{}_'.format(
                       criterion) + 'features_{}.sav'.format(max_features)
    p_?_test = test_sets_base_folder + '\\LSTM_RF_OCSVM\\{}_{}_{}\\?_test_{}_{}_{}_{}_{}'.format(
            folder_name, name, number_of_bins, desc, injection_length,
            step_over, percentage, epsilon)

    # 1. call get_models_folders_data_folders(RF_train_config)
    # 2. for each models folder (specifies data version and binning):
        # a. for each LSTM model(=model_folder) in the models' folder:
               model_name = model_folder + '_RF
               go to RFs\\diff_ + models_folder:
                go over all files and select the file which have model_name in their name.
                injection sets folder=models_folder + '_' + model_folder.split(_)[-1]
                d = get_desc(data version used)
                # b. for each injection params combination:
                    get p_x_test, p_y_test, p_labels according to injection params, d, injection sets folder.
                    run lstm on p_x_test, check if it's a diff model. if no then pass prediction to RF, else
                    pass the diff from y_test to each RF. (remove the first 20 labels from labels (check if it needs to be done))
                    compute metrics and write parameters and scores to file.
    """
    # 1.
    """with open(models_train_config, mode='r') as train_config:
        models_folders, data_folders, binning_dict, params = get_models_folders_data_folders(train_config)
        for i in range(len(models_folders)):
            models_folders[i] = group + '_' + models_folders[i]
        for i in range(len(data_folders)):
            data_folders[i] = group + '_' + data_folders[i]

    with open(tests_config_path, mode='r') as test_config:
        test_params = yaml.load(test_config, Loader=yaml.FullLoader)['processing_config']"""

    binning_methods, numbers_of_bins, data_versions = get_LSTM_params(lstm_config)
    injection_lengths, step_overs, percentages, epsilons = get_injection_params(injection_config)

    with open(ocsvm_config, mode='r') as config:
        params = yaml.load(config, Loader=yaml.FullLoader)['params_dict']

    kernels = params['kernel']
    nus = params['nu']

    folders = {"k_means": 'KMeans', "equal_frequency": 'EqualFreq',
               "equal_width": 'EqualWidth'}

    # 2.

    results_df = pd.DataFrame(columns=excel_cols)
    labels_df = pd.DataFrame(
        columns=['data version', 'binning', '# bins', 'kernel', 'nu', '# std count', 'injection length',
                 'percentage',
                 'window size',
                 '# window', 'model label', 'true label'])
    for data_version in data_versions:
        # a.
        folder_name = data_version['name']
        file_name = data_version['desc']
        for binning_method in binning_methods:
            bin_part = folders[binning_method]
            for number_of_bins in numbers_of_bins:
                validation_X = LSTM_validation + '//{}_{}//X_{}_{}_{}'.format(folder_name, folders[binning_method],
                                                                              file_name, binning_method, number_of_bins)
                validation_Y = LSTM_validation + '//{}_{}//y_{}_{}_{}'.format(folder_name, folders[binning_method],
                                                                              file_name, binning_method,
                                                                              number_of_bins)

                with open(validation_X, mode='rb') as val_p:
                    val_X = pickle.load(val_p)

                with open(validation_Y, mode='rb') as val_p:
                    val_y = pickle.load(val_p)

                model_name = '{}_{}_{}'.format(data_version['desc'], binning_method, number_of_bins)
                dump_model = data.modeles_path + '//{}_{}'.format(bin_part, folder_name)
                model_path = dump_model + '//' + model_name
                LSTM = keras.models.load_model(model_path)

                val_pred = LSTM.predict(val_X)

                model_name = model_name + '_OCSVM'
                classifiers_dir = models.SCADA_base + '//SVM//{}_{}'.format(bin_part, folder_name)
                for kernel in kernels:
                    for nu in nus:
                        # now get the exact svm model.
                        classifier = model_name + '_nu_{}_kernel_{}.sav'.format(nu, kernel)
                        with open(classifiers_dir + '//' + classifier,
                                  mode='rb') as classifier_p:
                            trained_classifier = pickle.load(classifier_p)

                        val_classifications = trained_classifier.predict(abs(val_pred - val_y))

                        # count -1 in windows.
                        # calc mean and std and set the threshold.
                        for window_size in window_sizes:
                            mean, std = count_outliers(val_classifications, window_size)
                            for num_std in nums_std:
                                count_threshold = mean + num_std * std

                                for injection_length in injection_lengths:
                                    for step_over in step_overs:
                                        anomaly_percentage = injection_length / (injection_length + step_over)
                                        if anomaly_percentage > 0.2:
                                            continue
                                        for percentage in percentages:
                                            for epsilon in epsilons:
                                                suffix = '{}_{}_{}_{}_{}_{}'.format(file_name,
                                                                                    binning_method,
                                                                                    number_of_bins,
                                                                                    injection_length,
                                                                                    step_over,
                                                                                    percentage)
                                                folder = '//LSTM//{}_{}_{}'.format(folder_name, binning_method,
                                                                                   number_of_bins)

                                                p_x_test = test_sets_base_folder + folder + '//X_test_' + suffix
                                                p_y_test = test_sets_base_folder + folder + '//y_test_' + suffix
                                                p_labels = test_sets_base_folder + folder + '//labels_' + suffix

                                                with open(p_x_test, mode='rb') as x_path:
                                                    X_test = pickle.load(x_path)
                                                with open(p_y_test, mode='rb') as y_path:
                                                    y_test = pickle.load(y_path)
                                                with open(p_labels, mode='rb') as l_path:
                                                    labels = pickle.load(l_path)

                                                pred = LSTM.predict(X_test)
                                                test = np.abs(pred - y_test)

                                                # make classifications.
                                                classifications = trained_classifier.predict(test)
                                                classifications = [1 if c == -1 else 0 for c in classifications]

                                                print("detected by OCSVM",
                                                      sum([1 if classifications[i - 20] == 1 and labels[i] == 1 else 0
                                                           for i in
                                                           range(20, len(classifications))]))

                                                true_windows_labels, model_windows_labels = LSTM_preds_to_window_preds(
                                                    classifications, labels, window_size, count_threshold)
                                                # detected, missed, mean_lag = measure_lag(model_windows_labels, labels, injection_length, step_over, w)
                                                labels_test_df = pd.DataFrame(columns=labels_df.columns)
                                                labels_test_df['# window'] = [i for i in
                                                                              range(len(true_windows_labels))]
                                                labels_test_df['window size'] = window_size
                                                labels_test_df['model label'] = model_windows_labels
                                                labels_test_df['true label'] = true_windows_labels
                                                labels_test_df['data version'] = data_version['name']
                                                labels_test_df['binning'] = bin_part
                                                labels_test_df['# bins'] = number_of_bins
                                                labels_test_df['injection length'] = injection_length
                                                labels_test_df['percentage'] = percentage
                                                labels_test_df['# std count'] = num_std
                                                with pd.ExcelWriter(xl_path, mode="a", engine="openpyxl",
                                                                    if_sheet_exists="overlay") as writer:
                                                    labels_test_df.to_excel(writer, sheet_name='windows labels')

                                                precision, recall, auc_score, f1, prc_auc_score, tn, fp, fn, tp = get_metrics(
                                                    true_windows_labels,
                                                    model_windows_labels)

                                                # parameters for excel.
                                                data_version_for_excel = data_version['name']
                                                binning_method_for_excel = binning_method
                                                number_of_bins_for_excel = number_of_bins

                                                result = {'data version': data_version_for_excel,
                                                          'binning': binning_method_for_excel,
                                                          '# bins': number_of_bins_for_excel, '# std count': num_std,
                                                          'window size': window_size, 'precision': precision,
                                                          'recall': recall, 'f1': f1,
                                                          'injection length': injection_length, 'step over': step_over,
                                                          'percentage': percentage, 'kernel': kernel, 'nu': nu}

                                                for col_name in excel_cols:
                                                    if col_name not in OCSVM_cols:
                                                        result[col_name] = '-'

                                                results_df = pd.concat([results_df,
                                                                        pd.DataFrame.from_dict(data={'0': result},
                                                                                               orient='index',
                                                                                               columns=excel_cols)],
                                                                       axis=0,
                                                                       ignore_index=True)
                                                with open(test_LSTM_OCSVM_log, mode='a') as test_log:
                                                    test_log.write('recorded results of OCSVM\n')
                                                    test_log.write(
                                                        'injection parameters are: len: {}, step: {}, %: {}, eps: {}\n'.format(
                                                            injection_length, step_over, percentage, epsilon))
                                                    test_log.write('model parameters:\n')
                                                    test_log.write(
                                                        'data version: {}, # bins: {}, binning method: {} ,# std:{}\n'.format(
                                                            data_version_for_excel, number_of_bins_for_excel,
                                                            binning_method_for_excel, num_std))
                                                    test_log.write('kernel: {}, nu: {}\n'.format(result['kernel'],
                                                                                                 result['nu']))
                                                    test_log.write(
                                                        'f1:{}, precision:{}, recall:{},tn:{}, fp:{}, fn:{}, tp:{}\n'.format(
                                                            f1,
                                                            precision,
                                                            recall, tn, fp, fn, tp))
    best_df = make_best(results_df)

    with pd.ExcelWriter(xl_path, mode="a", engine="openpyxl", if_sheet_exists="replace") as writer:
        best_df['algorithm'] = 'OCSVM'
        results_df['algorithm'] = 'OCSVM'
        sheet = 'OCSVM performance.'
        # if group != '':
        # sheet += 'group: {}'.format(group)
        results_df.to_excel(excel_writer=writer, sheet_name=sheet)
        sheet = 'OCSVM best scores.'
        # if group != '':
        # sheet += 'group: {}'.format(group)
        best_df.to_excel(excel_writer=writer, sheet_name=sheet)


#################################IRRELEVANT FOR NOW#####################################################


def test_FSTM(models_train_config, injection_config, tests_config_path, group=''):
    return None


#################################IRRELEVANT FOR NOW#####################################################
def test_LSTM_based_classifiers_many_PLCs(models_train_config, injection_config, tests_config_path, groups_ids):
    with open(injection_config, mode='r') as injection_conf:
        injection_params = yaml.load(injection_conf, Loader=yaml.FullLoader)
    with open(models_train_config, mode='r') as train_config:
        models_folders, data_folders, binning_dict, params = get_models_folders_data_folders(train_config)
        params = yaml.load(train_config, Loader=yaml.FullLoader)['bin_number']
        data_versions = params['data_version']
        bin_range = params['bin_number']
    for group_id in groups_ids:
        # this creates the sheet for each group.
        test_LSTM_based_classifiers(models_train_config, injection_config, tests_config_path, group_id)

    # Decide on the best parameter configuration. Use a weighted average of the scores.
    test_base = test_sets_base_folder
    test_sets_lengths = {}
    weights = {}
    total_samples = 0

    injection_lengths = injection_params['InjectionLength']
    step_overs = injection_params['StepOver']
    percentages = injection_params['Percentage']
    epsilons = injection_params['Epsilon']

    binning_methods = binning_dict.keys()
    bin_start = bin_range['start']
    bin_end = bin_range['end'] + 1

    # read test sets lengths.
    for group_id in groups_ids:
        with open(test_base + '\\group_{}'.format(group_id), mode='rb') as test_path:
            test_df = pickle.load(test_path)
            total_samples += len(test_df)
            test_sets_lengths[group_id] = len(test_df)

    # assign weights to each group.
    for group_id in groups_ids:
        from_group = test_sets_lengths[group_id]
        weight = from_group / total_samples
        weights[group_id] = weight

    for model_type in ['OCSVM', 'RF']:
        for prefix in ['', 'diff_']:
            sheets_dfs = {}
            best_df = pd.DataFrame(columns=best_cols)
            for group_id in groups_ids:
                sheet = prefix + model_type + ' best scores. group {}'.format(group_id)
                sheets_dfs[group_id] = pd.read_excel(xl_path, sheet)
            # now, get the metric scores for each injection x models configuration combination.
            for injection_length in injection_lengths:
                for step_over in step_overs:
                    anomaly_percentage = injection_length / (injection_length + step_over)
                    if anomaly_percentage > 0.2:
                        continue
                    for percentage in percentages:
                        for epsilon in epsilons:
                            for data_version in data_versions:
                                for bins in range(bin_start, bin_end):
                                    for binning_method in binning_methods:
                                        # now calculate the scores.
                                        method_f1 = 0
                                        method_auc = 0
                                        method_recall = 0
                                        method_precision = 0
                                        for group_id in groups_ids:
                                            group_best = sheets_dfs[group_id]
                                            mask = group_best['data_version'] == data_version and group_best[
                                                '# bins'] == bins \
                                                   and group_best['binning'] == binning_method and group_best[
                                                       'injection length'] == injection_length and group_best[
                                                       'step over'] == step_over

                                            # get the relevant entry in the df.
                                            matching = group_best.loc[mask]
                                            f1 = matching['f1']
                                            precision = matching['precision']
                                            auc_score = matching['auc score']
                                            recall = matching['recall']

                                            # update group scores.
                                            group_w = weights[group_id]
                                            method_auc += auc_score * group_w
                                            method_f1 += f1 * group_w
                                            method_recall += recall * group_w
                                            method_precision += precision * group_w

                                        # add entry to best_df
                                        new_entry = {'data version': data_version, 'binning': binning_method,
                                                     '# bins': bins,
                                                     'precision': method_precision, 'recall': method_recall,
                                                     'auc': method_auc, 'f1': method_f1,
                                                     'injection length': injection_length,
                                                     'step over': step_over, 'percentage': percentage}
                                        entry_df = pd.DataFrame.from_dict(columns=best_df.columns,
                                                                          data={'0': new_entry}, orient='index')
                                        best_df = pd.concat([best_df, entry_df], ignore_index=True)

            # write to xl.
            with pd.ExcelWriter(xl_path) as writer:
                sheet = model_type + prefix + ':many PLCs, best scores'
                best_df.to_excel(writer, sheet_name=sheet)


def test_LSTM_based_OCSVM(lstm_config, ocsvm_config, injection_config, group_info=None, effected_plcs_suffixes=None):
    """
    injection sets folder: folder_name, name, number_of_bins = binning method, data version, # bins
    path to RF = SCADA_BASE + '\\RFs\\' + 'diff_' + models_folder + '\\' + diff_' + model_name + 'estimators_{}_'.format(
                       estimators) + 'criterion{}_'.format(
                       criterion) + 'features_{}.sav'.format(max_features)
    p_?_test = test_sets_base_folder + '\\LSTM_RF_OCSVM\\{}_{}_{}\\?_test_{}_{}_{}_{}_{}'.format(
            folder_name, name, number_of_bins, desc, injection_length,
            step_over, percentage, epsilon)

    # 1. call get_models_folders_data_folders(RF_train_config)
    # 2. for each models folder (specifies data version and binning):
        # a. for each LSTM model(=model_folder) in the models' folder:
               model_name = model_folder + '_RF
               go to RFs\\diff_ + models_folder:
                go over all files and select the file which have model_name in their name.
                injection sets folder=models_folder + '_' + model_folder.split(_)[-1]
                d = get_desc(data version used)
                # b. for each injection params combination:
                    get p_x_test, p_y_test, p_labels according to injection params, d, injection sets folder.
                    run lstm on p_x_test, check if it's a diff model. if no then pass prediction to RF, else
                    pass the diff from y_test to each RF. (remove the first 20 labels from labels (check if it needs to be done))
                    compute metrics and write parameters and scores to file.
    """
    # 1.
    binning_methods, numbers_of_bins, data_versions = get_LSTM_params(lstm_config)
    injection_lengths, step_overs, percentages, epsilons = get_injection_params(injection_config)

    with open(ocsvm_config, mode='r') as config:
        params = yaml.load(config, Loader=yaml.FullLoader)['params_dict']

    kernels = params['kernel']
    nus = params['nu']

    folders = {"k_means": 'KMeans', "equal_frequency": 'EqualFreq',
               "equal_width": 'EqualWidth'}

    for data_version in data_versions:
        # a.
        folder_name = data_version['name']
        file_name = data_version['desc']
        for binning_method in binning_methods:
            bin_part = folders[binning_method]
            for number_of_bins in numbers_of_bins:
                validation_X = LSTM_validation + '//{}_{}//X_{}_{}_{}'.format(folder_name, folders[binning_method],
                                                                              file_name, binning_method, number_of_bins)
                validation_Y = LSTM_validation + '//{}_{}//y_{}_{}_{}'.format(folder_name, folders[binning_method],
                                                                              file_name, binning_method,
                                                                              number_of_bins)
                if group_info is not None:
                    validation_X = LSTM_validation + '//{}_{}//{}_X_{}_{}_{}'.format(group_info, folder_name,
                                                                                     folders[binning_method],
                                                                                     file_name, binning_method,
                                                                                     number_of_bins)
                    validation_Y = LSTM_validation + '//{}_{}//{}_y_{}_{}_{}'.format(group_info, folder_name,
                                                                                     folders[binning_method],
                                                                                     file_name, binning_method,
                                                                                     number_of_bins)

                with open(validation_X, mode='rb') as val_p:
                    val_X = pickle.load(val_p)

                with open(validation_Y, mode='rb') as val_p:
                    val_y = pickle.load(val_p)

                model_name = '{}_{}_{}'.format(data_version['desc'], binning_method, number_of_bins)
                dump_model = data.modeles_path + '//{}_{}'.format(bin_part, folder_name)

                if group_info is not None:
                    dump_model = data.modeles_path + '//{}_{}_{}'.format(group_info, bin_part, folder_name)

                model_path = dump_model + '//' + model_name
                LSTM = keras.models.load_model(model_path)

                val_pred = LSTM.predict(val_X)

                model_name = model_name + '_OCSVM'
                classifiers_dir = models.SCADA_base + '//SVM//{}_{}'.format(bin_part, folder_name)
                for kernel in kernels:
                    for nu in nus:
                        # now get the exact svm model.
                        classifier = f'{model_name}_nu_{nu}_kernel{kernel}.sav'
                        if group_info is not None:
                            classifier = f'{group_info}_{model_name}_nu_{nu}_kernel{kernel}.sav'

                        # now get the exact svm model.
                        with open(classifiers_dir + '//' + classifier,
                                  mode='rb') as classifier_p:
                            trained_classifier = pickle.load(classifier_p)

                        val_classifications = trained_classifier.predict(abs(val_pred - val_y))
                        val_classification_transformed = [1 if c == -1 else 0 for c in val_classifications]

                        # count -1 in windows.
                        # calc mean and std and set the threshold.
                        for window_size in window_sizes:
                            mean, std = count_outliers(val_classifications, window_size)
                            for num_std in nums_std:
                                count_threshold = mean + num_std * std

                                for injection_length in injection_lengths:
                                    for step_over in step_overs:
                                        anomaly_percentage = injection_length / (injection_length + step_over)
                                        if anomaly_percentage > 0.2:
                                            continue
                                        for percentage in percentages:
                                            for epsilon in epsilons:
                                                for effected_plcs_suffix in effected_plcs_suffixes:
                                                    suffix = '{}_{}_{}_{}_{}_{}'.format(file_name,
                                                                                        binning_method,
                                                                                        number_of_bins,
                                                                                        injection_length,
                                                                                        step_over,
                                                                                        percentage)
                                                    folder = '//LSTM//{}_{}_{}'.format(folder_name, binning_method,
                                                                                       number_of_bins)

                                                    p_x_test = test_sets_base_folder + folder + '//X_test_' + suffix
                                                    p_y_test = test_sets_base_folder + folder + '//y_test_' + suffix
                                                    p_labels = test_sets_base_folder + folder + '//labels_' + suffix

                                                    if group_info is not None:
                                                        p_x_test = test_sets_base_folder + folder + f'//{group_info}_X_test_' + suffix
                                                        p_y_test = test_sets_base_folder + folder + f'//{group_info}_y_test_' + suffix
                                                        p_labels = test_sets_base_folder + folder + f'//{group_info}_labels_' + suffix

                                                    if effected_plcs_suffix is not None:
                                                        p_x_test += f'_{effected_plcs_suffix}'
                                                        p_y_test += f'_{effected_plcs_suffix}'
                                                        p_labels += f'_{effected_plcs_suffix}'

                                                    with open(p_x_test, mode='rb') as x_path:
                                                        X_test = pickle.load(x_path)
                                                    with open(p_y_test, mode='rb') as y_path:
                                                        y_test = pickle.load(y_path)
                                                    with open(p_labels, mode='rb') as l_path:
                                                        labels = pickle.load(l_path)

                                                    pred = LSTM.predict(X_test)
                                                    test = np.abs(pred - y_test)

                                                    # make classifications.
                                                    classifications = trained_classifier.predict(test)
                                                    classifications = [1 if c == -1 else 0 for c in classifications]

                                                    true_windows_labels, model_windows_labels = LSTM_preds_to_window_preds(
                                                        classifications, labels, window_size, count_threshold)
                                                    # detected, missed, mean_lag = measure_lag(model_windows_labels, labels, injection_length, step_over, w)
                                                    """labels_test_df = pd.DataFrame(columns=labels_df.columns)
                                                    labels_test_df['# window'] = [i for i in
                                                                                  range(len(true_windows_labels))]
                                                    labels_test_df['window size'] = window_size
                                                    labels_test_df['model label'] = model_windows_labels
                                                    labels_test_df['true label'] = true_windows_labels
                                                    labels_test_df['data version'] = data_version['name']
                                                    labels_test_df['binning'] = bin_part
                                                    labels_test_df['# bins'] = number_of_bins
                                                    labels_test_df['injection length'] = injection_length
                                                    labels_test_df['percentage'] = percentage
                                                    labels_test_df['# std count'] = num_std
    
                                                    excel_path = paths_dict[data_version['name']]
    
                                                    with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl",
                                                                        if_sheet_exists="overlay") as writer:
                                                        row = 0
                                                        sheet_name = 'LSTM-OCSVM_{}_{}_{} windows labels'.format(
                                                            data_version['name'], bin_part, number_of_bins)
                                                        if group_info is not None:
                                                            sheet_name = '{}_LSTM-OCSVM_{}_{}_{} windows labels'.format(
                                                            group_info, data_version['name'], bin_part, number_of_bins)
                                                        if sheet_name in writer.sheets.keys():
                                                            row = writer.sheets[sheet_name].max_row
                                                        labels_test_df.to_excel(writer, sheet_name=sheet_name, startrow=row)"""

                                                    precision, recall, auc_score, f1, prc_auc_score, tn, fp, fn, tp = get_metrics(
                                                        true_windows_labels,
                                                        model_windows_labels)

                                                    excel_path = paths_dict[data_version['name']]

                                                    # parameters for excel.
                                                    data_version_for_excel = data_version['name']
                                                    binning_method_for_excel = binning_method
                                                    number_of_bins_for_excel = number_of_bins

                                                    result = {'group': group_info,
                                                              'injection type': effected_plcs_suffix if effected_plcs_suffix else 'all_plcs',
                                                              'data version': data_version_for_excel,
                                                              'binning': binning_method_for_excel,
                                                              '# bins': number_of_bins_for_excel,
                                                              '# std count': num_std,
                                                              'window size': window_size, 'precision': precision,
                                                              'recall': recall, 'f1': f1,
                                                              'injection length': injection_length,
                                                              'step over': step_over,
                                                              'percentage': percentage, 'kernel': kernel, 'nu': nu}

                                                    for col_name in excel_cols:
                                                        if col_name not in OCSVM_cols:
                                                            result[col_name] = '-'

                                                    res_df = pd.DataFrame.from_dict(data={'0': result},
                                                                                    columns=excel_cols,
                                                                                    orient='index')

                                                    row = 0

                                                    with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl",
                                                                        if_sheet_exists="overlay") as writer:
                                                        if 'performance' in writer.sheets.keys():
                                                            row = writer.sheets['performance'].max_row
                                                        res_df['algorithm'] = 'LSTM-OCSVM'
                                                        h = True
                                                        if row > 0:
                                                            h = False
                                                        res_df.to_excel(excel_writer=writer, sheet_name='performance',
                                                                        startrow=row, header=h, index=False)

                                                    with open(test_LSTM_OCSVM_log, mode='a') as test_log:
                                                        test_log.write('recorded results of OCSVM\n')
                                                        test_log.write(
                                                            'injection parameters are: len: {}, step: {}, %: {}, eps: {}, type:{}\n'.format(
                                                                injection_length, step_over, percentage, epsilon,
                                                                effected_plcs_suffix if effected_plcs_suffix else 'all_plcs'))
                                                        test_log.write('model parameters:\n')
                                                        test_log.write(
                                                            'data version: {}, # bins: {}, binning method: {} ,# std:{}, window: {}\n'.format(
                                                                data_version_for_excel, number_of_bins_for_excel,
                                                                binning_method_for_excel, num_std, window_size))
                                                        test_log.write('kernel: {}, nu: {}\n'.format(result['kernel'],
                                                                                                     result['nu']))
                                                        test_log.write(
                                                            'f1:{}, precision:{}, recall:{},tn:{}, fp:{}, fn:{}, tp:{}\n'.format(
                                                                f1,
                                                                precision,
                                                                recall, tn, fp, fn, tp))
                                                        test_log.write(
                                                            'OCSVM detected in test: {}, in val: {}\n'.format(
                                                                sum(classifications),
                                                                sum(val_classification_transformed)))
                                                        test_log.write(
                                                            'mean is {} std is {} threshold is {}\n'.format(mean, std,
                                                                                                            count_threshold))


def test_LSTM_based_OCSVM_plcs_split(lstm_config, ocsvm_config, injection_config, group_pool=data.active_ips,
                                     split_type='single_plc', effected_plcs_suffixes=None, injection_type='all_plcs',
                                     sheet_name='average_performance'):
    # 1. test every single PLC.
    """for group in group_pool:
        group_info = group
        test_LSTM_based_OCSVM(lstm_config, ocsvm_config, injection_config, group_info=group_info, effected_plcs_suffixes=effected_plcs_suffixes)"""

    # 2. find the weight of the results of each PLC.
    total_length = 0
    sub_groups_in_split_weights = {sub_groups_in_split: 0 for sub_groups_in_split in group_pool}

    for group in group_pool:
        raw_test_set_path = data.datasets_path + f'\\{split_type}\\{group}_test'
        if 'single_plc' in group:
            group_name = group[11:]
            raw_test_set_path = data.datasets_path + f'\\{split_type}\\{group_name}_test'
        if split_type == 'all_plcs':
            raw_test_set_path = data.datasets_path + f'\\{split_type}\\test'

        with open(raw_test_set_path, mode='rb') as raw_test_set_f:
            raw_test_set = pickle.load(raw_test_set_f)
        raw_test_set_length = len(raw_test_set)
        total_length += raw_test_set_length
        sub_groups_in_split_weights[group] = raw_test_set_length

    for sub_groups_in_split in sub_groups_in_split_weights.keys():
        sub_groups_in_split_weights[sub_groups_in_split] /= total_length

    # 3. calculate weighted average.
    results_df = pd.read_excel(xl_path2, sheet_name='performance')
    metric_cols = ['f1', 'precision', 'recall']
    other_cols = results_df.columns[:-3]
    total_df = pd.DataFrame(columns=excel_cols)

    for effected_plcs_suffix in effected_plcs_suffixes:
        averaged_results_df = None
        for group in group_pool:
            group_name = group
            sub_groups_in_split_weight = sub_groups_in_split_weights[group]
            results_mask = (results_df['group'] == group_name) & (
                    results_df['injection type'] == (effected_plcs_suffix if effected_plcs_suffix else 'all_plcs'))
            sub_groups_in_split_res = results_df.loc[results_mask, metric_cols] * sub_groups_in_split_weight
            sub_groups_in_split_res = data.reset_df_index(sub_groups_in_split_res)

            if averaged_results_df is None:
                averaged_results_df = sub_groups_in_split_res
                averaged_results_df = data.reset_df_index(averaged_results_df)
            else:
                averaged_results_df = averaged_results_df.add(sub_groups_in_split_res, fill_value=0)
                averaged_results_df = data.reset_df_index(averaged_results_df)

        effected_plcs_suffix_total_df = pd.concat(
            [results_df.iloc[:(len(averaged_results_df)), :-3], averaged_results_df], axis=1,
            ignore_index=False)
        effected_plcs_suffix_total_df = effected_plcs_suffix_total_df.drop('group', axis=1)

        effected_plcs_suffix_total_df = data.reset_df_index(effected_plcs_suffix_total_df)

        total_df = pd.concat([total_df, effected_plcs_suffix_total_df], axis=0, ignore_index=True)
        total_df = data.reset_df_index(total_df)

    # 4. update excel file.
    with pd.ExcelWriter(xl_path2, mode="a", engine="openpyxl",
                        if_sheet_exists="overlay") as writer:
        row = 0
        if sheet_name in writer.sheets.keys():
            row = writer.sheets[sheet_name + "_std0"].max_row
        total_df['algorithm'] = 'LSTM-OCSVM'
        total_df['split type'] = split_type
        total_df['injection type'] = injection_type
        total_df["# std count"] = 0
        h = True
        if row > 0:
            h = False
        total_df.to_excel(excel_writer=writer, sheet_name=sheet_name,
                          startrow=row, header=h, index=False)


def test_LSTM_based_OCSVM_all_plcs_split(lstm_config, ocsvm_config, injection_config):
    # 1. test. no need to average anything like in the previous case.
    group_info = 'all_plcs'
    test_LSTM_based_OCSVM(lstm_config, ocsvm_config, injection_config, group_info)


def test_LSTM_OCSVM_all_splits_all_injections(lstm_config, ocsvm_config, injection_config):
    for split_type in SPLIT_TYPES:
        for injection_type, effected_plcs_suffixes in injection_type_to_effected_plcs_suffixes.items():
            test_LSTM_based_OCSVM_plcs_split(lstm_config, ocsvm_config, injection_config, split_type=split_type,
                                             effected_plcs_suffixes=effected_plcs_suffixes,
                                             group_pool=SPLIT_TYPE_TO_GROUP_POOL[split_type],
                                             injection_type=injection_type)


#################################IRRELEVANT FOR NOW#####################################################

def load_val_data_and_dict(binning_method, bins):
    val_path = val_base + '//DFA'
    with open(val_path + '//X_{}_{}'.format(binning_method, bins), mode='rb') as df_f:
        val_data = pickle.load(df_f)
    with open(val_path + '//dict_{}_{}'.format(binning_method, bins), mode='rb') as dict_f:
        val_pkts_to_states_dict = pickle.load(dict_f)
    return val_data, val_pkts_to_states_dict


def paramterize_dfa_detections(val_decisions, w, pkts_to_states_dict):
    counts = []
    for i in range(len(val_decisions) - w + 1):
        window_start_state = pkts_to_states_dict[i]
        window_end_state = pkts_to_states_dict[i + w - 1]
        count = sum(val_decisions[window_start_state: window_end_state])
        counts.append(count)
    return calc_MLE_mean(counts), cal_MLE_std(counts)


def test_DFA(injection_config, group=None, effected_plcs_suffixes=None):
    folders = {"k_means": 'KMeans', "equal_frequency": 'EqualFreq',
               "equal_width": 'EqualWidth'}
    n_bins, binners, names = get_DFA_params()
    n_bins = [3, 4, 5, 6, 7]
    injection_lengths, step_overs, percentages, epsilons = get_injection_params(injection_config)
    binning_methods_names = {'KMeans': "k_means", 'EqualFreq': "equal_frequency", 'EqualWidth': "equal_width"}

    """results_df = pd.DataFrame(columns=excel_cols)
        labels_df = pd.DataFrame(
        columns=['binning', '# bins', '# std count', 'injection length', 'step over', 'percentage', 'window size',
                 '# window', 'model label', 'true label'])"""

    for binner in binners:
        for bins in n_bins:
            with open(data.automaton_path + "//{}_{}".format(folders[binner], bins), mode='rb') as dfa_path:
                DFA = pickle.load(dfa_path)
            val_data, val_pkts_to_states_dict = load_val_data_and_dict(binner, bins)

            val_decisions, new_state_val, bad_time_val, new_t_val = models.automaton.detect(DFA, val_data,
                                                                                            val_data.columns[2:])

            for injection_length in injection_lengths:
                for step_over in step_overs:
                    anomaly_percentage = injection_length / (injection_length + step_over)
                    if anomaly_percentage > 0.2:
                        continue
                    for percentage in percentages:
                        for epsilon in epsilons:
                            for effected_plcs_suffix in effected_plcs_suffixes:
                                p_x_test = test_sets_base_folder + '//DFA//{}_X_test_{}_{}_{}_{}_{}'.format(
                                    group,
                                    binner,
                                    bins,
                                    injection_length,
                                    step_over,
                                    percentage)
                                p_labels = test_sets_base_folder + '//DFA//{}_labels_{}_{}_{}_{}_{}'.format(
                                    group,
                                    binner,
                                    bins,
                                    injection_length,
                                    step_over,
                                    percentage)
                                p_pkt_dict = test_sets_base_folder + '//DFA//{}_dict_{}_{}_{}_{}_{}'.format(
                                    group, binner, bins, injection_length,
                                    step_over,
                                    percentage)
                                parent = test_sets_base_folder + '//raw'
                                labels_path = parent + '//{}_labels_{}_{}_{}'.format(group, injection_length, step_over,
                                                                                     percentage)

                                if effected_plcs_suffix is not None:
                                    p_x_test += f'_{effected_plcs_suffix}'
                                    p_labels += f'_{effected_plcs_suffix}'
                                    p_pkt_dict += f'_{effected_plcs_suffix}'

                                with open(p_x_test, mode='rb') as test_data_path:
                                    test_df = pickle.load(test_data_path)
                                with open(p_labels, mode='rb') as labels_p:
                                    test_labels = pickle.load(labels_p)
                                with open(p_pkt_dict, mode='rb') as pkts_to_stats_p:
                                    pkts_to_states_dict = pickle.load(pkts_to_stats_p)
                                with open(labels_path, mode='rb') as labels_f:
                                    raw_test_labels = pickle.load(labels_f)

                                # the DFA classifies transitions.
                                registers = test_df.columns[2:]
                                start = time.time()
                                decisions, new_state, bad_time, new_t = models.automaton.detect(DFA, test_df, registers)
                                end = time.time()

                                elapsed = end - start
                                avg_elapsed = elapsed / len(test_labels)

                                for w in window_sizes:
                                    mean, std = paramterize_dfa_detections(val_decisions, w, val_pkts_to_states_dict)
                                    for num_std in nums_std:
                                        count_threshold = mean + num_std * std
                                        true_windows_labels, dfa_window_labels = DFA_window_labels(decisions,
                                                                                                   raw_test_labels, w,
                                                                                                   pkts_to_states_dict,
                                                                                                   count_threshold)
                                        precision, recall, auc_score, f1, prc_auc_score, tn, fp, fn, tp = get_metrics(
                                            y_true=true_windows_labels, y_pred=dfa_window_labels)

                                        result = {'group': group,
                                                  'injection type': effected_plcs_suffix if effected_plcs_suffix is not None else 'all_plcs',
                                                  'binning': binning_methods_names[names[binner]],
                                                  '# bins': bins,
                                                  '# std count': num_std,
                                                  'injection length': injection_length,
                                                  'window size': w,
                                                  'step over': step_over,
                                                  'percentage': percentage,
                                                  'precision': precision,
                                                  'recall': recall,
                                                  'f1': f1, }

                                        """labels_test_df = pd.DataFrame(columns=labels_df.columns)
                                        labels_test_df['# window'] = [i for i in range(len(true_windows_labels))]
                                        labels_test_df['window size'] = w
                                        labels_test_df['model label'] = dfa_window_labels
                                        labels_test_df['true label'] = true_windows_labels
                                        labels_test_df['binning'] = binning_methods_names[names[binner]]
                                        labels_test_df['# bins'] = bins
                                        labels_test_df['# std count'] = num_std
                                        labels_test_df['injection length'] = injection_length
                                        labels_test_df['step over'] = step_over
                                        labels_test_df['percentage'] = percentage
                                        with pd.ExcelWriter(xl_path, mode="a", engine="openpyxl",
                                                            if_sheet_exists="overlay") as writer:
                                            row = 0
                                            sheet_name = 'DFA_{}_{} windows labels'.format(
                                                binning_methods_names[names[binner]], bins)
                                            if sheet_name in writer.sheets.keys():
                                                row = writer.sheets[sheet_name].max_row
                                            labels_test_df.to_excel(writer, sheet_name=sheet_name, startrow=row)"""

                                        for col_name in excel_cols:
                                            if col_name not in DFA_cols:
                                                result[col_name] = '-'

                                        res_df = pd.DataFrame.from_dict(data={'0': result}, columns=excel_cols,
                                                                        orient='index')

                                        with pd.ExcelWriter(xl_path, mode="a", engine="openpyxl",
                                                            if_sheet_exists="overlay") as writer:
                                            res_df['algorithm'] = 'DFA'

                                            row = 0

                                            if 'performance' in writer.sheets.keys():
                                                row = writer.sheets['performance'].max_row

                                            h = True
                                            if row > 0:
                                                h = False
                                            res_df.to_excel(excel_writer=writer, sheet_name='performance', startrow=row,
                                                            header=h, index=False)

                                        with open(test_DFA_log, mode='a') as test_log:
                                            test_log.write(
                                                f'recorded DFA{group} results for injection with parameters:\n')
                                            test_log.write(
                                                'inference: {}, avg inference: {}, binning: {}, # bins: {}, window: {}, # std count: {}, injection type:{}\n'.format(
                                                    elapsed,
                                                    avg_elapsed,
                                                    names[
                                                        binner],
                                                    bins, w, num_std,
                                                    effected_plcs_suffix if effected_plcs_suffix is not None else 'all plcs'))
                                            test_log.write(
                                                'len: {}, step: {}, %: {}\n'.format(injection_length, step_over,
                                                                                    percentage))
                                            test_log.write(
                                                'scores: precision: {}, recall: {}, f1: {}, tn: {}, fn : {}, tp: {}, fp: {}\n'.format(
                                                    result['precision'],
                                                    result['recall'],
                                                    result['f1'], tn, fn, tp, fp))
                                            test_log.write(
                                                'new_state:{}, bad_time:{}, new_t:{}\n'.format(new_state, bad_time,
                                                                                               new_t))
                                            test_log.write(
                                                '(in val) new_state:{}, bad_time:{}, new_t:{}\n'.format(new_state_val,
                                                                                                        bad_time_val,
                                                                                                        new_t_val))


def test_DFA_plcs_split(injection_config, group_pool=data.active_ips, split_type='single_plc',
                        sheet_name='average_performance', effected_plcs_suffixes=None, injection_type='all_plcs'):
    """

    :param injection_config: injection parameters.
    :param group_pool: the groups of plcs in the split (spearman_0 spearman_1) or (k_means_0, k_means_1) etc.
    :param split_type: textual description of the split (all_plcs, single_plc, spearman, etc.)
    :param sheet_name: the sheet name (will always be average_performance)
    :param effected_plcs_suffixes: describes the injection types.
    :return:
    """
    # 1. test every single PLC.
    """for group in group_pool:
        group_info = group
        test_DFA(injection_config, group=group_info, effected_plcs_suffixes=effected_plcs_suffixes)"""

    # 2. find the weight of the results of each PLC.
    total_length = 0
    sub_groups_in_split_weights = {sub_groups_in_split: 0 for sub_groups_in_split in group_pool}

    for group in group_pool:
        raw_test_set_path = data.datasets_path + f'\\{split_type}\\{group}_test'
        if 'single_plc' in group:
            group_name = group[11:]
            raw_test_set_path = data.datasets_path + f'\\{split_type}\\{group_name}_test'
        if split_type == 'all_plcs':
            raw_test_set_path = data.datasets_path + f'\\{split_type}\\test'

        with open(raw_test_set_path, mode='rb') as raw_test_set_f:
            raw_test_set = pickle.load(raw_test_set_f)
        raw_test_set_length = len(raw_test_set)
        total_length += raw_test_set_length
        sub_groups_in_split_weights[group] = raw_test_set_length

    for sub_groups_in_split in sub_groups_in_split_weights.keys():
        sub_groups_in_split_weights[sub_groups_in_split] /= total_length

    # 3. calculate weighted average.
    results_df = pd.read_excel(xl_path, sheet_name='performance')
    metric_cols = ['f1', 'precision', 'recall']
    other_cols = results_df.columns[:-3]
    total_df = pd.DataFrame(columns=excel_cols)

    for effected_plcs_suffix in effected_plcs_suffixes:
        averaged_results_df = None
        for group in group_pool:
            group_name = group
            sub_groups_in_split_weight = sub_groups_in_split_weights[group]
            results_mask = (results_df['group'] == group_name) & (
                    results_df['injection type'] == (effected_plcs_suffix if effected_plcs_suffix else 'all_plcs'))
            sub_groups_in_split_res = results_df.loc[results_mask, metric_cols] * sub_groups_in_split_weight
            sub_groups_in_split_res = data.reset_df_index(sub_groups_in_split_res)

            print('printing sub_groups_in_split_res')
            print(sub_groups_in_split_res)

            if averaged_results_df is None:
                averaged_results_df = sub_groups_in_split_res
                averaged_results_df = data.reset_df_index(averaged_results_df)
            else:
                averaged_results_df = averaged_results_df.add(sub_groups_in_split_res, fill_value=0)
                averaged_results_df = data.reset_df_index(averaged_results_df)

            print('printing averaged results df')
            print(averaged_results_df, len(averaged_results_df))

        effected_plcs_suffix_total_df = pd.concat(
            [results_df.iloc[:(len(averaged_results_df)), :-3], averaged_results_df], axis=1,
            ignore_index=False)  # CAN NOT DO THIS!!!!!!!

        effected_plcs_suffix_total_df = data.reset_df_index(effected_plcs_suffix_total_df)
        print('printing effected_plcs_suffix_total_df')
        print(effected_plcs_suffix_total_df)

        total_df = pd.concat([total_df, effected_plcs_suffix_total_df], axis=0, ignore_index=True)
        total_df = data.reset_df_index(total_df)
        print('printing total_df')
        print(total_df)

        total_df = pd.concat([total_df, effected_plcs_suffix_total_df], axis=0, ignore_index=True)

    # 4. update excel file.
    with pd.ExcelWriter(xl_path, mode="a", engine="openpyxl",
                        if_sheet_exists="overlay") as writer:
        row = 0

        if sheet_name in writer.sheets.keys():
            row = writer.sheets[sheet_name].max_row

        total_df['split type'] = split_type
        total_df['injection type'] = injection_type
        total_df = total_df.drop('group', axis=1)

        h = True
        if row > 0:
            h = False

        total_df.to_excel(excel_writer=writer, sheet_name=sheet_name,
                          startrow=row, header=h, index=False)


def test_DFA_all_plcs_split(injection_config):
    test_DFA(injection_config, group='all_plcs')


def test_DFA_all_splits_all_injections(injection_config):
    for split_type in SPLIT_TYPES:
        print(f'SPLIT : {split_type}')
        for injection_type, effected_plcs_suffixes in injection_type_to_effected_plcs_suffixes.items():
            print(f'INJECTION TYPE: {injection_type}')
            test_DFA_plcs_split(injection_config, split_type=split_type, effected_plcs_suffixes=effected_plcs_suffixes,
                                group_pool=SPLIT_TYPE_TO_GROUP_POOL[split_type], injection_type=injection_type)
            break
        break


"""def test_LSTM_OCSVM_all_splits_all_injections(lstm_config, ocsvm_config, injection_config):
    for split_type in SPLIT_TYPES:
        for injection_type, effected_plcs_suffixes in injection_type_to_effected_plcs_suffixes.items():
            test_LSTM_based_OCSVM_plcs_split(lstm_config, ocsvm_config, injection_config, split_type=split_type,
                                             effected_plcs_suffixes=effected_plcs_suffixes,
                                             group_pool=SPLIT_TYPE_TO_GROUP_POOL[split_type],
                                             injection_type=injection_type)"""


######################IRRELEVANT FOR NOW########################################
# functions for training LSTMs.
def train_LSTM(train_config, train_df=None, group_info=None):
    """
    for each data version * bins * binning method:
        process raw data
        train lstm
    :param train_config:
    :return:
    """
    folders = {"k_means": 'KMeans', "equal_frequency": 'EqualFreq',
               "equal_width": 'EqualWidth'}
    # lstm params.
    methods, bins, data_versions = get_LSTM_params(train_config)

    if train_df is None:
        raw_df = data.load(data.datasets_path, 'TRAIN')
    else:
        raw_df = train_df

    for data_version in data_versions:
        folder_name = data_version['name']
        file_name = data_version['desc']
        processed = None
        if not data_version['reprocess']:
            processed = data.process(raw_df, folder_name, None, None, False)
        for number_of_bins in bins:
            for method in methods:
                method_folder = folders[method]
                method_name = data_version['desc']

                model_name = '{}_{}_{}'.format(file_name, method_name, number_of_bins)
                if group_info is None:
                    suffix = '//{}_{}'.format(method_folder, folder_name) + '//{}'.format(model_name)
                else:
                    suffix = '//{}_{}'.format(method_folder, folder_name) + '//{}_{}'.format(group_info, model_name)
                binners_p = binners_base + '//{}_{}'.format(method_folder, folder_name)
                scalers_p = scalers_base + '//{}_{}'.format(method_folder, folder_name)

                if not os.path.exists(binners_p):
                    Path(binners_p).mkdir(exist_ok=True, parents=True)

                if not os.path.exists(scalers_p):
                    Path(scalers_p).mkdir(exist_ok=True, parents=True)

                if data_version['reprocess']:
                    lstm_input = data.process(raw_df, folder_name, number_of_bins, method_name, True,
                                              binner_path=suffix)
                else:
                    lstm_input = processed.copy()
                    cols_not_to_bin = data_version['no_bin']

                    # scale everything, bin by config file.
                    for col_name in lstm_input.columns:
                        if 'time' not in col_name and 'state' not in col_name and col_name not in cols_not_to_bin:
                            data.bin_col(lstm_input, method, col_name, number_of_bins, path=suffix)
                        lstm_input[col_name] = data.scale_col(lstm_input, col_name, path=suffix)

                if group_info is None:
                    dump_model = data.modeles_path + '\\{}_{}'.format(method_folder, folder_name)
                    dump_df = data.datasets_path + '\\{}_{}'.format(method_folder, folder_name)
                else:
                    dump_model = data.modeles_path + '\\{}_{}_{}'.format(group_info, method_folder, folder_name)
                    dump_df = data.datasets_path + '\\{}_{}_{}'.format(group_info, method_folder, folder_name)

                models.models.simple_LSTM(lstm_input, 20, 42, model_name, train=1.0, models_path=dump_model,
                                          data_path=dump_df)


def train_LSTM_single_PLCs(train_config):
    load_path = data.datasets_path + '//single_plc//'
    ips = data.active_ips

    for ip in ips:
        with open(load_path + f'{ip}_train', mode='rb') as train_f:
            plc_df = pickle.load(train_f)
        with open(LSTM_train_log, mode='a') as log:
            log.write(f'training LSTMs, single PLCs split, PLC: {ip}')
        train_LSTM(train_config, train_df=plc_df, group_info=f'single_plc_{ip}')


def train_LSTM_all_PLCs(train_config):
    load_path = data.datasets_path + '//all_plcs//train'

    with open(load_path, mode='rb') as train_f:
        all_plcs_df = pickle.load(train_f)
    with open(LSTM_train_log, mode='a') as log:
        log.write('training LSTMs, all PLCs split\n')
    train_LSTM(train_config, train_df=all_plcs_df, group_info='all plcs')


def train_LSTM_pearson_split(train_config):
    for group_name in pearson_groups:
        load_path = data.datasets_path + f'//pearson//{group_name}'
        with open(load_path, mode='rb') as train_f:
            pearson_df = pickle.load(train_f)
        with open(LSTM_train_log, mode='a') as log:
            log.write('training LSTMs, pearson split\n')
        train_LSTM(train_config, train_df=pearson_df, group_info=group_name)


def train_LSTM_spearman_split(train_config):
    for group_name in pearson_groups:
        load_path = data.datasets_path + f'//spearman//{group_name}'
        with open(load_path, mode='rb') as train_f:
            spearman_df = pickle.load(train_f)
        with open(LSTM_train_log, mode='a') as log:
            log.write('training LSTMs, spearman split\n')
        train_LSTM(train_config, train_df=spearman_df, group_info=group_name)


def train_LSTM_k_means_split(train_config):
    for group_name in pearson_groups:
        load_path = data.datasets_path + f'//k_means_split//{group_name}'
        with open(load_path, mode='rb') as train_f:
            k_means_df = pickle.load(train_f)
        with open(LSTM_train_log, mode='a') as log:
            log.write('training LSTMs, k_means split\n')
        train_LSTM(train_config, train_df=k_means_df, group_info=group_name)


# for LSTM classifiers.
def create_test_sets_LSTMs(train_config, injection_config, group_info=None, df_for_regs=None,
                           effected_plcs_suffix=None):
    folders = {"k_means": 'KMeans', "equal_frequency": 'EqualFreq',
               "equal_width": 'EqualWidth'}

    lim = 0.2  # don't allow more than 20 percent of malicious packets in the data set.

    with open(injection_config, mode='r') as anomalies_config:
        injection_params = yaml.load(anomalies_config, Loader=yaml.FullLoader)
        injection_lengths = injection_params['InjectionLength']
        step_overs = injection_params['StepOver']
        percentages = injection_params['Percentage']
        epsilons = injection_params['Epsilon']

    # first ,inject anomalies. and create the test set for: LSTM , RF and OCSVM.
    for injection_length in injection_lengths:
        for step_over in step_overs:
            anomaly_percentage = injection_length / (injection_length + step_over)
            if anomaly_percentage > lim:
                pass
            else:
                for percentage in percentages:
                    for epsilon in epsilons:
                        df_path = test_sets_base_folder + '//raw//{}_data_{}_{}_{}'.format(group_info,
                                                                                           injection_length,
                                                                                           step_over,
                                                                                           percentage)
                        labels_path = test_sets_base_folder + '//raw//{}_labels_{}_{}_{}'.format(group_info,
                                                                                                 injection_length,
                                                                                                 step_over,
                                                                                                 percentage)
                        if effected_plcs_suffix is not None:
                            df_path += f'_{effected_plcs_suffix}'
                            labels_path += f'_{effected_plcs_suffix}'

                        with open(df_path, mode='rb') as p:
                            anomalous_data = pickle.load(p)

                        with open(labels_path, mode='rb') as p:
                            labels = pickle.load(p)

                        #  process the data using the different versions
                        with open(train_config, mode='r') as processing_config:
                            config = yaml.load(processing_config, Loader=yaml.FullLoader)
                            binnings = config['binning_methods']
                            data_versions = config['train_sets_config']
                            numbers_of_bins = config['bins']

                        for method_name in binnings:
                            folder_name = folders[method_name]
                            # folder_name: name of binning method in the folders (KMeans), method_name: name of binning method in files (kmeans)
                            for data_version in data_versions:
                                name = data_version['name']
                                desc = data_version['desc']
                                if not data_version['reprocess']:
                                    processed = data.process(anomalous_data, name, None, None, False,
                                                             df_for_regs=df_for_regs)
                                for number_of_bins in numbers_of_bins:
                                    if data_version['reprocess']:
                                        lstm_input = data.process(anomalous_data, folder_name, number_of_bins,
                                                                  method_name,
                                                                  True, df_for_regs=df_for_regs)
                                    else:
                                        lstm_input = processed.copy()
                                        cols_not_to_bin = data_version['no_bin']
                                        method_folder = folders[method_name]
                                        file_name = data_version['desc']
                                        folder_name = data_version['name']

                                        model_name = '{}_{}_{}'.format(file_name, method_name, number_of_bins)
                                        if group_info is None:
                                            suffix = '//{}_{}'.format(method_folder, folder_name) + '//{}'.format(
                                                model_name)
                                        else:
                                            group = group_info
                                            if group_info is 'all_plcs':
                                                group = 'all plcs'
                                            suffix = '//{}_{}'.format(method_folder, folder_name) + '//{}_{}'.format(
                                                group,
                                                model_name)

                                        # scale everything, bin by config file.
                                        for col_name in lstm_input.columns:
                                            lstm_input[col_name] = lstm_input[col_name].fillna(
                                                lstm_input[col_name].mean())

                                            if 'time' not in col_name and 'state' not in col_name and col_name not in cols_not_to_bin:
                                                with open(binners_base + suffix + '_{}'.format(col_name),
                                                          mode='rb') as binner_p:
                                                    binner = pickle.load(binner_p)

                                                lstm_input[col_name] = binner.transform(
                                                    lstm_input[col_name].to_numpy().reshape(-1, 1))

                                            with open(scalers_base + suffix + '_{}'.format(col_name),
                                                      mode='rb') as scaler_f:
                                                scaler = pickle.load(scaler_f)

                                            lstm_input[col_name] = scaler.transform(
                                                lstm_input[col_name].to_numpy().reshape(-1, 1))

                                    # now create test data set for LSTM. Only need X_test and y_test.
                                    X_test, y_test = models.custom_train_test_split(
                                        lstm_input,
                                        20, 42, train=1.0)
                                    # now save, X_test, y_test and the labels which will be used to obtain the y_test of the classifier.
                                    p_suffix = '{}_{}_{}_{}_{}_{}'.format(
                                        file_name, method_name, number_of_bins, injection_length,
                                        step_over, percentage)

                                    if effected_plcs_suffix is not None:
                                        p_suffix += f'_{effected_plcs_suffix}'

                                    # make sure dirs exist and dump.
                                    dir_path = test_sets_base_folder + '//LSTM//{}_{}_{}'.format(
                                        folder_name, method_name, number_of_bins)

                                    if group_info is not None:
                                        p_x_test = dir_path + f'//{group_info}_X_test_' + p_suffix
                                        p_y_test = dir_path + f'//{group_info}_y_test_' + p_suffix
                                        p_labels = dir_path + f'//{group_info}_labels_' + p_suffix
                                    else:
                                        p_x_test = dir_path + '//X_test_' + p_suffix
                                        p_y_test = dir_path + '//y_test_' + p_suffix
                                        p_labels = dir_path + '//labels_' + p_suffix

                                    if not os.path.exists(dir_path):
                                        Path(dir_path).mkdir(parents=True, exist_ok=True)

                                    with open(p_x_test, mode='wb') as data_path:
                                        pickle.dump(X_test, data_path)
                                    with open(p_y_test, mode='wb') as data_path:
                                        pickle.dump(y_test, data_path)
                                    with open(p_labels, mode='wb') as data_path:
                                        pickle.dump(labels, data_path)


def create_LSTM_test_sets_single_plc_split(train_config, injection_config, effected_plcs_suffix=None):
    for ip in data.active_ips:
        df_for_regs_path = data.datasets_path + f'//single_plc//{ip}_train'
        with open(df_for_regs_path, mode='rb') as df_f:
            df_for_regs = pickle.load(df_f)
        create_test_sets_LSTMs(train_config, injection_config, f'single_plc_{ip}', df_for_regs=df_for_regs,
                               effected_plcs_suffix=effected_plcs_suffix)


def create_LSTM_test_sets_all_plcs_split(train_config, injection_config, effected_plcs_suffix=None):
    df_for_regs_path = data.datasets_path + f'//all_plcs//train'
    with open(df_for_regs_path, mode='rb') as df_f:
        df_for_regs = pickle.load(df_f)
    create_test_sets_LSTMs(train_config, injection_config, 'all_plcs', df_for_regs=df_for_regs,
                           effected_plcs_suffix=effected_plcs_suffix)


def create_LSTM_test_sets_pearson_split(train_config, injection_config, effected_plcs_suffix=None):
    for group in pearson_groups:
        df_for_regs_path = data.datasets_path + f'//pearson//{group}_train'
        with open(df_for_regs_path, mode='rb') as df_f:
            df_for_regs = pickle.load(df_f)
        create_test_sets_LSTMs(train_config, injection_config, group, df_for_regs=df_for_regs,
                               effected_plcs_suffix=effected_plcs_suffix)


def create_LSTM_test_sets_spearman_split(train_config, injection_config, effected_plcs_suffix=None):
    for group in spearman_groups:
        df_for_regs_path = data.datasets_path + f'//spearman//{group}_train'
        with open(df_for_regs_path, mode='rb') as df_f:
            df_for_regs = pickle.load(df_f)
        create_test_sets_LSTMs(train_config, injection_config, group, df_for_regs=df_for_regs,
                               effected_plcs_suffix=effected_plcs_suffix)


def create_LSTM_test_sets_k_means_split(train_config, injection_config, effected_plcs_suffix=None):
    for group in k_means_groups:
        df_for_regs_path = data.datasets_path + f'//k_means//{group}_train'
        with open(df_for_regs_path, mode='rb') as df_f:
            df_for_regs = pickle.load(df_f)
        create_test_sets_LSTMs(train_config, injection_config, group, df_for_regs=df_for_regs,
                               effected_plcs_suffix=effected_plcs_suffix)


def create_all_LSTM_test_sets_with_injections_by_splits(train_config, injection_config):
    for effected_plcs, effected_plcs_suffix in effected_plcs_data_tuples:
        create_LSTM_test_sets_single_plc_split(train_config=train_config, injection_config=injection_config,
                                               effected_plcs_suffix=effected_plcs_suffix)
        create_LSTM_test_sets_all_plcs_split(train_config=train_config, injection_config=injection_config,
                                             effected_plcs_suffix=effected_plcs_suffix)
        create_LSTM_test_sets_spearman_split(train_config=train_config, injection_config=injection_config,
                                             effected_plcs_suffix=effected_plcs_suffix)
        create_LSTM_test_sets_pearson_split(train_config=train_config, injection_config=injection_config,
                                            effected_plcs_suffix=effected_plcs_suffix)
        create_LSTM_test_sets_k_means_split(train_config=train_config, injection_config=injection_config,
                                            effected_plcs_suffix=effected_plcs_suffix)


def test_LSTM(train_config, group_info=None, df_for_regs=None):
    folders = {"k_means": 'KMeans', "equal_frequency": 'EqualFreq',
               "equal_width": 'EqualWidth'}

    # load test data.
    if group_info != "all_plcs":
        test_df = data.load(test_sets_base_folder, data.datasets_path + f"//{group_info}_test")
    else:
        test_df = data.load(test_sets_base_folder, data.datasets_path + "//test")

    cols = ['data_version', 'binning', '# bins', 'mse', 'r2'] if group_info is None else \
        ['group', 'data_version', 'binning', '# bins', 'mse', 'r2']

    results_df = pd.DataFrame(columns=cols)

    # go over all combinations, process raw test set, test and save metric scores.
    with open(train_config, mode='r') as c:
        train_params = yaml.load(c, Loader=yaml.FullLoader)

    binning_methods = train_params['binning_methods']
    numbers_of_bins = train_params['bins']
    data_versions = train_params['train_sets_config']

    for data_version in data_versions:
        test_lstm = None
        file_name = data_version['desc']
        folder_name = data_version['name']
        if not data_version['reprocess']:
            test_lstm = data.process(test_df, data_version['name'], None, None, False, df_for_regs=df_for_regs,
                                     registers=None)
        for binning_method in binning_methods:
            method_folder = folders[binning_method]
            for number_of_bins in numbers_of_bins:
                if data_version['reprocess']:
                    model_name = '{}_{}_{}'.format(file_name, binning_method, number_of_bins)
                    suffix = '//{}_{}'.format(method_folder, folder_name) + '//{}'.format(model_name)

                    if group_info is not None:
                        suffix = '//{}_{}'.format(method_folder, folder_name) + '//{}_{}'.format(
                            group_info if group_info is not "all_plcs" else "all plcs"
                            , model_name)

                    lstm_in = data.process(test_df, data_version['name'], number_of_bins, binning_method, True,
                                           binner_path=suffix, df_for_regs=df_for_regs)
                else:
                    lstm_in = test_lstm.copy()
                    cols_not_to_bin = data_version['no_bin']

                    model_name = '{}_{}_{}'.format(file_name, binning_method, number_of_bins)
                    suffix = '//{}_{}'.format(method_folder, folder_name) + '//{}'.format(model_name)

                    if group_info is not None:
                        suffix = '//{}_{}'.format(method_folder, folder_name) + '//{}_{}'.format(
                            group_info if group_info is not "all_plcs" else "all plcs"
                            , model_name)

                    # scale everything, bin by config file.
                    for col_name in lstm_in.columns:
                        lstm_in[col_name] = lstm_in[col_name].fillna(lstm_in[col_name].mean())

                        if 'time' not in col_name and 'state' not in col_name and col_name not in cols_not_to_bin:
                            with open(binners_base + suffix + '_{}'.format(col_name), mode='rb') as binner_p:
                                binner = pickle.load(binner_p)

                            lstm_in[col_name] = binner.transform(lstm_in[col_name].to_numpy().reshape(-1, 1))

                        with open(scalers_base + suffix + '_{}'.format(col_name), mode='rb') as scaler_f:
                            scaler = pickle.load(scaler_f)

                        lstm_in[col_name] = scaler.transform(lstm_in[col_name].to_numpy().reshape(-1, 1))

                bin_part = folders[binning_method]
                version_part = data_version['name']
                model_name = '{}_{}_{}'.format(data_version['desc'], binning_method, number_of_bins)
                dump_model = data.modeles_path + '//{}_{}'.format(bin_part, version_part)

                if group_info is not None:
                    dump_model = data.modeles_path + '//{}_{}_{}'.format(group_info, bin_part, version_part)
                    if group_info == "all_plcs":
                        dump_model = data.modeles_path + '//{}_{}_{}'.format("all plcs", bin_part, version_part)

                LSTM = keras.models.load_model(dump_model + '//' + model_name)

                X_test, y_test = models.models.custom_train_test_split(lstm_in, 20, 42, train=1.0)
                y_pred = LSTM.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)

                result = {'data_version': data_version['name'],
                          'binning': folders[binning_method],
                          '# bins': number_of_bins,
                          'mse': mse,
                          'r2': r2}

                if group_info is not None:
                    result['group'] = group_info

                res_df = pd.DataFrame.from_dict(columns=results_df.columns, data={'0': result}, orient='index')
                results_df = pd.concat([results_df, res_df], ignore_index=True)
                with open(lstm_test_log, mode='a') as log:
                    log.write('mse:{}, r2:{}, version:{}, binning:{}, bins:{}, samples:{}\n'.format(mse, r2,
                                                                                                    data_version[
                                                                                                        'name'],
                                                                                                    folders[
                                                                                                        binning_method],
                                                                                                    number_of_bins,
                                                                                                    len(X_test)))

    with pd.ExcelWriter(xl_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
        row = 0
        sheet = 'LSTM scores names'
        if sheet in writer.sheets.keys():
            row = writer.sheets[sheet].max_row
        header = row == 0
        results_df.to_excel(writer, sheet_name='LSTM scores names', startrow=row, header=header)


def test_LSTM_single_plc_split(train_config):
    for active_ip in data.active_ips:
        group_info = f'single_plc_{active_ip}'
        df_for_regs_path = data.datasets_path + f'//single_plc//{active_ip}_train'

        with open(df_for_regs_path, mode='rb') as df_f:
            df_for_regs = pickle.load(df_f)

        test_LSTM(train_config, group_info=group_info, df_for_regs=df_for_regs)


def test_LSTM_all_plcs_split(train_config):
    group_info = f'all_plcs'

    df_for_regs_path = data.datasets_path + f'//all_plcs//train'
    with open(df_for_regs_path, mode='rb') as df_f:
        df_for_regs = pickle.load(df_f)

    test_LSTM(train_config, group_info=group_info, df_for_regs=df_for_regs)


def test_LSTM_spearman_split(train_config):
    for group in spearman_groups:
        df_for_regs_path = data.datasets_path + f'//spearman//{group}_train'

        with open(df_for_regs_path, mode='rb') as df_f:
            df_for_regs = pickle.load(df_f)

        test_LSTM(train_config, group_info=group, df_for_regs=df_for_regs)


def test_LSTM_pearson_split(train_config):
    for group in pearson_groups:
        df_for_regs_path = data.datasets_path + f'//pearson//{group}_train'

        with open(df_for_regs_path, mode='rb') as df_f:
            df_for_regs = pickle.load(df_f)

        test_LSTM(train_config, group_info=group, df_for_regs=df_for_regs)


def test_LSTM_k_means_split(train_config):
    for group in k_means_groups:
        df_for_regs_path = data.datasets_path + f'//k_means//{group}_train'

        with open(df_for_regs_path, mode='rb') as df_f:
            df_for_regs = pickle.load(df_f)

        test_LSTM(train_config, group_info=group, df_for_regs=df_for_regs)


def test_all_LSTM_regressors(train_config):
    test_LSTM_single_plc_split(train_config)
    test_LSTM_all_plcs_split(train_config)
    test_LSTM_pearson_split(train_config)
    test_LSTM_spearman_split(train_config)
    test_LSTM_k_means_split(train_config)


def detect_LSTM(lstm_config, injection_config, d='L2', t='# std'):
    injection_lengths, step_overs, percentages, injection_epsilons = get_injection_params(injection_config)

    # grid over the lstm data sets : get labels and data.
    # for each data set: predict and get metrics.

    folders = {"k_means": 'KMeans', "equal_frequency": 'EqualFreq',
               "equal_width": 'EqualWidth'}

    results_df = pd.DataFrame(columns=excel_cols)
    labels_df = pd.DataFrame(
        columns=['data version', 'binning', '# bins', '#std', 'p_value', 'distance metric', 'injection length',
                 'percentage',
                 'window size',
                 '# window', 'model label', 'true label'])

    # go over all combinations, process raw test set, test and save metric scores.
    binning_methods, numbers_of_bins, data_versions = get_LSTM_params(lstm_config)

    # for each lstm (grid over lstm_config):
    # get the validation data set, predict on it, calculate the deviation mean + 3 std.
    for data_version in data_versions:
        folder_name = data_version['name']
        file_name = data_version['desc']
        for binning_method in binning_methods:
            for number_of_bins in numbers_of_bins:
                validation_X = LSTM_validation + '{}_{}\\X_{}_{}_{}'.format(folder_name, folders[binning_method],
                                                                            file_name, binning_method, number_of_bins)
                validation_Y = LSTM_validation + '{}_{}\\y_{}_{}_{}'.format(folder_name, folders[binning_method],
                                                                            file_name, binning_method,
                                                                            number_of_bins)

                with open(validation_X, mode='rb') as val_p:
                    val_X = pickle.load(val_p)

                with open(validation_Y, mode='rb') as val_p:
                    val_y = pickle.load(val_p)

                bin_part = folders[binning_method]
                version_part = data_version['name']
                model_name = '{}_{}_{}'.format(data_version['desc'], binning_method, number_of_bins)
                dump_model = data.modeles_path + '\\{}_{}'.format(bin_part, version_part)
                model_path = dump_model + '\\' + model_name
                LSTM = keras.models.load_model(model_path)

                pred = LSTM.predict(val_X)

                if d == 'MD':
                    data_for_cov = np.concatenate([val_X[0], val_y])  # covariance of validation data.
                    cov_val = np.cov(np.transpose(data_for_cov))
                    cov_val_inv = np.linalg.inv(cov_val)
                    deviations = calc_MD(pred, val_y,
                                         cov_val_inv)  # Mahalanobis distance of prediction from ground truth.
                elif d == 'L1':
                    deviations = calc_L1(pred, val_y)
                else:
                    deviations = calc_L2(pred, val_y)

                val_size = len(val_X)
                mean = calc_MLE_mean(deviations)
                std = cal_MLE_std(deviations)

                limits = None
                if t == 'std':
                    limits = nums_std
                else:
                    limits = p_values

                for limit in limits:
                    if t == 'std':
                        threshold = mean + std * limit
                        val_labels = [1 if dist > threshold else 0 for dist in deviations]
                    else:
                        val_labels = [1 if calc_p_value(mean, std, dist) < limit else 0 for dist in deviations]

                    for injection_length in injection_lengths:
                        for step_over in step_overs:
                            anomaly_percentage = injection_length / (injection_length + step_over)
                            if anomaly_percentage > lim:
                                pass
                            else:
                                for percentage in percentages:
                                    for epsilon in injection_epsilons:
                                        suffix = '{}_{}_{}_{}_{}_{}'.format(file_name,
                                                                            binning_method,
                                                                            number_of_bins,
                                                                            injection_length,
                                                                            step_over,
                                                                            percentage)
                                        folder = '//LSTM//{}_{}_{}'.format(folder_name, binning_method, number_of_bins)

                                        p_x_test = test_sets_base_folder + folder + '//X_test_' + suffix
                                        p_y_test = test_sets_base_folder + folder + '//y_test_' + suffix
                                        p_labels = test_sets_base_folder + folder + '//labels_' + suffix

                                        with open(p_x_test, mode='rb') as x_path:
                                            X_test = pickle.load(x_path)
                                        with open(p_y_test, mode='rb') as y_path:
                                            Y_test = pickle.load(y_path)
                                        with open(p_labels, mode='rb') as l_path:
                                            labels = pickle.load(l_path)

                                        test_pred = LSTM.predict(X_test)

                                        if d == 'MD':
                                            data_for_cov_test = np.concatenate(
                                                [X_test[0], Y_test])  # covariance of validation data.
                                            cov_val_test = np.cov(np.transpose(data_for_cov_test))
                                            cov_test_inv = np.linalg.inv(cov_val_test)

                                            test_deviations = calc_MD(test_pred, Y_test, cov_test_inv)
                                        elif d == 'L1':
                                            test_deviations = calc_L1(test_pred, Y_test)
                                        else:
                                            test_deviations = calc_L2(test_pred, Y_test)

                                        if t == '# std':
                                            pred_labels = [1 if dist > threshold else 0 for dist in test_deviations]
                                        else:
                                            pred_labels = [1 if calc_p_value(mean, std, dist) < limit else 0 for dist in
                                                           test_deviations]

                                        for w in window_sizes:
                                            above_thresh_counts = []
                                            for i in range(val_size - w + 1):
                                                count = sum(val_labels[i: i + w])
                                                above_thresh_counts.append(count)

                                            count_mean = calc_MLE_mean(above_thresh_counts)
                                            count_std = cal_MLE_std(above_thresh_counts)

                                            for num_std_count in num_stds_count:
                                                count_threshold = count_mean + count_std * num_std_count
                                                true_windows_labels, model_windows_labels = LSTM_preds_to_window_preds(
                                                    pred_labels, labels, w, count_threshold)
                                                labels_test_df = pd.DataFrame(columns=labels_df.columns)
                                                labels_test_df['# window'] = [i for i in
                                                                              range(len(true_windows_labels))]
                                                labels_test_df['window size'] = w
                                                labels_test_df['model label'] = model_windows_labels
                                                labels_test_df['true label'] = true_windows_labels
                                                labels_test_df['data version'] = data_version['name']
                                                labels_test_df['binning'] = bin_part
                                                labels_test_df['# bins'] = number_of_bins
                                                labels_test_df['injection length'] = injection_length
                                                labels_test_df['percentage'] = percentage
                                                if t == '# std':
                                                    labels_test_df['# std'] = limit
                                                    labels_test_df['p_value'] = '-'
                                                else:
                                                    labels_test_df['# std'] = '-'
                                                    labels_test_df['p_value'] = limit
                                                labels_test_df['distance metric'] = d
                                                labels_test_df['# std count'] = num_std_count
                                                labels_df = pd.concat([labels_df, labels_test_df], ignore_index=True)

                                                precision, recall, auc_score, f1, prc_auc_score = get_metrics(
                                                    true_windows_labels,
                                                    model_windows_labels)

                                                result = {'data version': data_version['name'],
                                                          'binning': bin_part,
                                                          '# bins': number_of_bins,
                                                          '# std': limit if t == '# std' else '-',
                                                          'p_value': limit if t == 'p_value' else '-',
                                                          'distance metric': d,
                                                          '# std count': num_std_count,
                                                          'window size': w,
                                                          'injection length': injection_length,
                                                          'percentage': percentage,
                                                          'precision': precision,
                                                          'recall': recall,
                                                          'auc': auc_score,
                                                          'f1': f1,
                                                          'prc': prc_auc_score}

                                                for col_name in excel_cols:
                                                    if col_name not in LSTM_detection_cols:
                                                        result[col_name] = '-'
                                                results_df = pd.concat(
                                                    [results_df,
                                                     pd.DataFrame.from_dict(data={'0': result}, columns=excel_cols,
                                                                            orient='index')],
                                                    axis=0, ignore_index=True)
                                                with open(test_LSTM_STD_log, mode='a') as test_log:
                                                    test_log.write(
                                                        'tested, data version:{}, binning:{}, #bins:{}, window: {}, {}: {}, # std count: {}, distance:{}\n'.format(
                                                            folder_name,
                                                            binning_method,
                                                            number_of_bins, w, t, limit, num_std_count, d))
                                                    test_log.write(
                                                        'injection: {}, {}, {}\n'.format(injection_length, step_over,
                                                                                         percentage))
                                                    test_log.write(
                                                        'auc:{}, f1:{}, prc:{}, precision:{}, recall:{}\n'.format(
                                                            auc_score, f1,
                                                            prc_auc_score,
                                                            precision,
                                                            recall))
    results_df['algorithm'] = 'LSTM+STD'
    with pd.ExcelWriter(xl_path) as writer:
        results_df.to_excel(writer, sheet_name='LSTM+STD scores')

    with pd.ExcelWriter(xl_labels_path) as writer:
        labels_df.to_excel(writer, sheet_name='LSTM window labels')


def create_val_for_LSTM(lstm_config, val_data_path=None, group_info=None):
    folders = {"k_means": 'KMeans', "equal_frequency": 'EqualFreq',
               "equal_width": 'EqualWidth'}
    if val_data_path is None:
        val_df = data.load(data.datasets_path, 'VAL')
    else:
        with open(val_data_path, mode='rb') as val_file:
            val_df = pickle.load(val_file)

    # go over all combinations, process raw test set, test and save metric scores.
    binning_methods, numbers_of_bins, data_versions = get_LSTM_params(lstm_config)

    for data_version in data_versions:
        val_lstm = None
        folder_name = data_version['name']
        file_name = data_version['desc']
        if not data_version['reprocess']:
            val_lstm = data.process(val_df, data_version['name'], None, None, False)
        for binning_method in binning_methods:
            method_folder = folders[binning_method]
            for number_of_bins in numbers_of_bins:
                if data_version['reprocess']:
                    lstm_in = data.process(val_df, number_of_bins, binning_method, True)
                else:
                    lstm_in = val_lstm.copy()
                    cols_not_to_bin = data_version['no_bin']
                    model_name = '{}_{}_{}'.format(file_name, binning_method, number_of_bins)
                    if val_data_path is None:
                        suffix = '//{}_{}'.format(method_folder, folder_name) + '//{}'.format(model_name)
                    else:
                        suffix = '//{}_{}'.format(method_folder, folder_name) + '//{}_{}'.format(group_info, model_name)

                    scaler = None

                    # scale everything, bin by config file.
                    for col_name in lstm_in.columns:
                        lstm_in[col_name] = lstm_in[col_name].fillna(lstm_in[col_name].mean())

                        if 'time' not in col_name and 'state' not in col_name and col_name not in cols_not_to_bin:
                            with open(binners_base + suffix + '_{}'.format(col_name), mode='rb') as binner_p:
                                binner = pickle.load(binner_p)

                            lstm_in[col_name] = binner.transform(lstm_in[col_name].to_numpy().reshape(-1, 1))

                        with open(scalers_base + suffix + '_{}'.format(col_name), mode='rb') as scaler_f:
                            scaler = pickle.load(scaler_f)

                        lstm_in[col_name] = scaler.transform(lstm_in[col_name].to_numpy().reshape(-1, 1))

                X_val, y_val = models.custom_train_test_split(lstm_in, series_len=20, np_seed=42, train=1.0)

                p_dir = LSTM_validation + '//{}_{}'.format(folder_name, folders[binning_method])
                if not os.path.exists(p_dir):
                    Path(p_dir).mkdir(parents=True, exist_ok=True)

                if group_info is None:
                    validation_path_X = p_dir + '//X_{}_{}_{}'.format(file_name, binning_method, number_of_bins)
                    validation_path_Y = p_dir + '//y_{}_{}_{}'.format(file_name, binning_method, number_of_bins)
                else:
                    validation_path_X = p_dir + '//{}_X_{}_{}_{}'.format(group_info, file_name, binning_method,
                                                                         number_of_bins)
                    validation_path_Y = p_dir + '//{}_y_{}_{}_{}'.format(group_info, file_name, binning_method,
                                                                         number_of_bins)
                with open(validation_path_X, mode='wb') as x_val_f:
                    pickle.dump(X_val, x_val_f)
                with open(validation_path_Y, mode='wb') as y_val_f:
                    pickle.dump(y_val, y_val_f)


def create_LSTM_val_single_plc_split(lstm_config):
    load_path = data.datasets_path + '//single_plc//'
    ips = data.active_ips

    for ip in ips:
        val_path = load_path + f'{ip}_val'
        group_info = f'single_plc_{ip}'
        create_val_for_LSTM(lstm_config, val_data_path=val_path, group_info=group_info)


def create_LSTM_val_all_plcs(lstm_config):
    val_path = data.datasets_path + '//all_plcs//val'
    group_info = 'all_plcs'
    create_val_for_LSTM(lstm_config, val_data_path=val_path, group_info=group_info)


def create_LSTM_val_pearson_split(lstm_config):
    val_path = data.datasets_path + '//all_plcs//val'

    for group in pearson_groups:
        create_val_for_LSTM(lstm_config, val_data_path=val_path, group_info=group)


def create_LSTM_val_spearman_split(lstm_config):
    val_path = data.datasets_path + '//all_plcs//val'

    for group in pearson_groups:
        create_val_for_LSTM(lstm_config, val_data_path=val_path, group_info=group)


def create_LSTM_val_k_means_split(lstm_config):
    val_path = data.datasets_path + '//all_plcs//val'

    for group in pearson_groups:
        create_val_for_LSTM(lstm_config, val_data_path=val_path, group_info=group)


def create_raw_test_sets(injections_config, test_df_path=None, group_info=None, plcs=data.active_ips,
                         effected_plcs=data.active_ips, effected_plcs_suffix=None):
    with open(injections_config, mode='r') as injections_conf:
        injection_params = yaml.load(injections_conf, Loader=yaml.FullLoader)

    injection_lengths = injection_params['InjectionLength']
    step_overs = injection_params['StepOver']
    percentages = injection_params['Percentage']
    epsilons = injection_params['Epsilon']
    lim = 0.2

    if test_df_path is None:
        with open(test_sets_base_folder + '//TEST', mode='rb') as test_path:
            test_data = pickle.load(test_path)
    else:
        with open(test_df_path, mode='rb') as test_path:
            test_data = pickle.load(test_path)

    for injection_length in injection_lengths:
        for step_over in step_overs:
            anomaly_percentage = injection_length / (injection_length + step_over)
            if anomaly_percentage > lim:
                pass
            else:
                for percentage in percentages:
                    for epsilon in epsilons:
                        anomalous_data, labels = inject_to_sub_group(test_data, injection_length, step_over,
                                                                     percentage,
                                                                     epsilon, effected_plcs)

                        df_path = test_sets_base_folder + '//raw//{}_data_{}_{}_{}'.format(group_info,
                                                                                           injection_length,
                                                                                           step_over,
                                                                                           percentage)
                        labels_path = test_sets_base_folder + '//raw//{}_labels_{}_{}_{}'.format(group_info,
                                                                                                 injection_length,
                                                                                                 step_over,
                                                                                                 percentage)
                        if effected_plcs_suffix is not None:
                            df_path += f'_{effected_plcs_suffix}'
                            labels_path += f'_{effected_plcs_suffix}'

                        plcs_mask = anomalous_data['src_ip'].isin(plcs) | anomalous_data['dst_ip'].isin(plcs)

                        with open(df_path, mode='wb') as df_p:
                            pickle.dump(data.reset_df_index(anomalous_data[plcs_mask]), df_p)

                        indices = anomalous_data.index[plcs_mask].tolist()

                        with open(labels_path, mode='wb') as labels_p:
                            pickle.dump(labels[indices], labels_p)


def create_raw_test_sets_sinple_plcs_split(injection_config, effected_plcs=data.active_ips, effected_plcs_suffix=None):
    ips = data.active_ips

    for ip in ips:
        test_data_path = data.datasets_path + '//all_plcs//test'
        group_info = f'single_plc_{ip}'
        if effected_plcs_suffix is None:
            create_raw_test_sets(injection_config, test_data_path, group_info, plcs=[ip])
        else:
            create_raw_test_sets(injection_config, test_data_path, group_info, plcs=[ip], effected_plcs=effected_plcs,
                                 effected_plcs_suffix=effected_plcs_suffix)


# THIS HAS TO HAPPEN FIRST.
def create_raw_test_sets_all_plcs_split(injection_config, effected_plcs=data.active_ips, effected_plcs_suffix=None):
    load_path = data.datasets_path + '//all_plcs//test'
    group_info = 'all_plcs'
    if effected_plcs_suffix is None:
        create_raw_test_sets(injection_config, load_path, group_info=group_info)
    else:
        create_raw_test_sets(injection_config, load_path, group_info=group_info, effected_plcs=effected_plcs,
                             effected_plcs_suffix=effected_plcs_suffix)


def create_raw_test_sets_pearson_split(injection_config, effected_plcs=data.active_ips, effected_plcs_suffix=None):
    for group in pearson_groups:
        load_path = data.datasets_path + '//all_plcs//test'
        if '0' in group:
            plcs = pearson_groups_ips[0]
        else:
            plcs = pearson_groups_ips[1]
        if effected_plcs_suffix is None:
            create_raw_test_sets(injection_config, load_path, group_info=group, plcs=plcs)
        else:
            create_raw_test_sets(injection_config, load_path, group_info=group, plcs=plcs,
                                 effected_plcs=effected_plcs, effected_plcs_suffix=effected_plcs_suffix)


def create_raw_test_sets_spearman_split(injection_config, effected_plcs=data.active_ips, effected_plcs_suffix=None):
    for group in pearson_groups:
        load_path = data.datasets_path + '//all_plcs//test'
        if '0' in group:
            plcs = spearman_groups_ips[0]
        else:
            plcs = spearman_groups_ips[1]
        if effected_plcs_suffix is None:
            create_raw_test_sets(injection_config, load_path, group_info=group, plcs=plcs)
        else:
            create_raw_test_sets(injection_config, load_path, group_info=group, plcs=plcs, effected_plcs=effected_plcs,
                                 effected_plcs_suffix=effected_plcs_suffix)


def create_raw_test_sets_k_means_split(injection_config, effected_plcs=data.active_ips, effected_plcs_suffix=None):
    for group in pearson_groups:
        load_path = data.datasets_path + '//all_plcs//test'
        if '0' in group:
            plcs = k_means_groups_ips[0]
        else:
            plcs = k_means_groups_ips[1]
        if effected_plcs_suffix is None:
            create_raw_test_sets(injection_config, load_path, group_info=group, plcs=plcs)
        else:
            create_raw_test_sets(injection_config, load_path, group_info=group, plcs=plcs, effected_plcs=effected_plcs,
                                 effected_plcs_suffix=effected_plcs_suffix)


def create_all_raw_test_sets_with_injections_by_splits(injection_config):
    for effected_plcs, effected_plcs_suffix in effected_plcs_data_tuples:
        create_raw_test_sets_sinple_plcs_split(injection_config, effected_plcs=effected_plcs,
                                               effected_plcs_suffix=effected_plcs_suffix)
        create_raw_test_sets_all_plcs_split(injection_config, effected_plcs=effected_plcs,
                                            effected_plcs_suffix=effected_plcs_suffix)
        create_raw_test_sets_pearson_split(injection_config, effected_plcs=effected_plcs,
                                           effected_plcs_suffix=effected_plcs_suffix)
        create_raw_test_sets_spearman_split(injection_config, effected_plcs=effected_plcs,
                                            effected_plcs_suffix=effected_plcs_suffix)
        create_raw_test_sets_k_means_split(injection_config, effected_plcs=effected_plcs,
                                           effected_plcs_suffix=effected_plcs_suffix)


# xl file will save for each model: model name, model parameters, injection parameters, true labels for window, model labels for windows.
# this is an additional file. other files for the metrics. true window labels and labels according to the models.
def LSTM_preds_to_window_preds(model_detections, true_labels, window_size, count_threshold):
    model_windows_labels = []
    true_windows_labels = []
    length = len(true_labels)  # true labels is labels of PACKETS (injected / benign).
    for i in range(length - window_size + 1):
        true_window_label = max(true_labels[i: i + window_size])
        if i < 21:
            # there are missing detections for part of the window. consider only labels for the packets
            # for which there are detections.
            # take labels for packets: i...i+window-1 = predictions from 0...i+window-1-21.
            # the jth prediction is for the j+21 packet (j >= 1)
            model_window_label = 1 if sum(model_detections[:i + window_size - 21]) > count_threshold else 0
        else:
            # take labels for packets: i-21...min(i+window-1-21, |detections|).
            # use min to avoid out of boundary exception.
            model_window_label = 1 if sum(
                model_detections[i - 21: min(i + window_size - 21, len(model_detections))]) > count_threshold else 0
        true_windows_labels.append(true_window_label)
        model_windows_labels.append(model_window_label)

    return true_windows_labels, model_windows_labels


def DFA_window_labels(decisions, true_labels, window_size, pkts_to_states_dict, threshold):
    dfa_windows_labels = []
    true_windows_labels = []
    length = len(true_labels)  # true labels is labels of PACKETS (injected / benign).
    for i in range(length - window_size + 1):
        true_window_label = max(true_labels[i: i + window_size])
        window_start_state = pkts_to_states_dict[i]
        window_end_state = pkts_to_states_dict[i + window_size - 1]
        dfa_window_label = 1 if sum(decisions[window_start_state: window_end_state]) > threshold else 0
        true_windows_labels.append(true_window_label)
        dfa_windows_labels.append(dfa_window_label)

    return true_windows_labels, dfa_windows_labels


# WHEN COMPARING METHODS TO KLSTM, NEED TO TAKE THE LABELS FROM THE 21ST WINDOW FOR THE OTHER METHOD.
def KLSTM_window_labels(model_windows_labels, true_labels, window_size, count_threshold):
    klstm_windows_labels = []
    true_windows_labels = []
    length = len(true_labels)  # true labels is labels of PACKETS (injected / benign).
    # KLSTM labels windows from the 21st window.
    for i in range(20, length - window_size + 1):
        true_window_label = max(true_labels[i: i + window_size])
        model_window_label = model_windows_labels[i - 20]  # window 21 = index 0 = first window ...
        klstm_windows_labels.append(model_window_label)
        true_windows_labels.append(true_window_label)

    return true_windows_labels, klstm_windows_labels


def FSTM_window_labels(model_windows_labels, true_labels, window_size, pkts_to_states_dict):
    fstm_windows_labels = []
    true_windows_labels = []
    length = len(true_labels)  # true labels is labels of PACKETS (injected / benign).
    # FSTM labels windows from the 21st window.

    for i in range(20, length - window_size + 1):
        true_window_label = max(true_labels[i: i + window_size])
        window_start_state = max(pkts_to_states_dict[i], 20)
        window_end_state = max(pkts_to_states_dict[i + window_size - 1], 20)
        fstm_window_label = max(model_windows_labels[window_start_state - 20: window_end_state + 1 - 20])
        fstm_windows_labels.append(fstm_window_label)
        true_windows_labels.append(true_window_label)

    return true_windows_labels, fstm_windows_labels


# functions to get parameters from files. refactored.
def get_KL_params(KL_config_path):
    with open(KL_config_path, mode='r') as KL_params_path:
        KL_params = yaml.load(KL_params_path, Loader=yaml.FullLoader)

    binning_methods = KL_params['binningmethods']
    bins = KL_params['bins']
    windows = KL_params['windows']
    epsilons = KL_params['epsilons']
    max_gaps = KL_params['maxgaps']
    k_values = KL_params['minhorizontalsups']
    default_hs = KL_params['defaulths']
    kernels = KL_params['kernel']
    nus = KL_params['nu']

    return binning_methods, bins, windows, epsilons, max_gaps, k_values, default_hs, kernels, nus


def get_DFA_params():
    bins = [4, 5, 6, 7]
    binning_methods = {'equal_width': data.equal_width_discretization}
    names = {'k_means': 'KMeans', 'equal_frequency': 'EqualFreq',
             'equal_width': 'EqualWidth'}
    return bins, binning_methods, names


def get_FSTM_params(FSTM_config):
    with open(FSTM_config, mode='r') as algo_config:
        FSTM_params = yaml.load(algo_config, Loader=yaml.FullLoader)

    time_windows = FSTM_params['window']
    min_supports = FSTM_params['supp']
    numbers_of_bins = FSTM_params['bins']
    binning_methods = FSTM_params['binning']
    series_lengths = FSTM_params['series']
    return time_windows, min_supports, numbers_of_bins, binning_methods, series_lengths


def get_LSTM_params(lstm_config):
    with open(lstm_config, mode='r') as c:
        train_params = yaml.load(c, Loader=yaml.FullLoader)

    binning_methods = train_params['binning_methods']
    numbers_of_bins = train_params['bins']
    data_versions = train_params['train_sets_config']
    return binning_methods, numbers_of_bins, data_versions


def get_injection_params(injections_config_path):
    with open(injections_config_path, mode='r') as injection_config:
        injection_params = yaml.load(injection_config, Loader=yaml.FullLoader)

    injection_lengths = injection_params['InjectionLength']
    step_overs = injection_params['StepOver']
    percentages = injection_params['Percentage']
    injection_epsilons = injection_params['Epsilon']
    return injection_lengths, step_overs, percentages, injection_epsilons


def get_metrics(y_true, y_pred):
    precisions, recalls, thresholds = precision_recall_curve(
        y_true=y_true,
        probas_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    auc_score = roc_auc_score(y_true=y_true, y_score=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    prc_auc_score = auc(recalls, precisions)
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()

    return precision, recall, auc_score, f1, prc_auc_score, tn, fp, fn, tp


def get_transitions_labels(anomalous_data, labels, algo_input):
    # labels of transitions.
    transitions_labels = []

    # time in state.
    time_in_state = 0

    # number of state
    state_idx = 0

    # packets to states dict. for the labeling of windows when testing.
    pkts_to_states = dict()
    pkts_to_states[0] = 0
    pkts_to_states[1] = 0

    # labels of the packets in the state.
    packet_labels_in_state = [labels[1]]
    # arrival time of the last known packet in the state.
    last_time = anomalous_data.loc[1, 'time']
    for pkt_idx in range(2, len(anomalous_data)):
        time_in_state += (
                anomalous_data.loc[pkt_idx, 'time'] - last_time).total_seconds()
        if time_in_state == algo_input.iloc[state_idx, 1]:
            time_in_state = 0
            state_idx += 1
            transitions_labels.append(max(packet_labels_in_state))
            packet_labels_in_state = [labels[pkt_idx]]
        else:
            packet_labels_in_state.append(labels[pkt_idx])
        pkts_to_states[pkt_idx] = state_idx
        last_time = anomalous_data.loc[pkt_idx, 'time']

    return transitions_labels, pkts_to_states


def calc_L2(pred, true):
    deviations = []
    for i in range(len(pred)):
        real = true[i]
        predicted = pred[i]
        dist = np.linalg.norm(real - predicted)
        deviations.append(dist)

    return deviations


def calc_MD(pred, val_y, cov_val_inv):
    MDs = []
    for i in range(len(pred)):
        MDs.append(distance.mahalanobis(pred[i], val_y[i], cov_val_inv))
    return MDs


def calc_L1(pred, true):
    distances = []
    for i in range(len(pred)):
        real = true[i]
        predicted = pred[i]
        dist = abs(real - predicted)
        distances.append(dist)
    return distances


def calc_MLE_mean(values):
    return np.mean(values)


def cal_MLE_std(values):
    mle_mean = calc_MLE_mean(values)
    s = 0
    for value in values:
        s += (value - mle_mean) ** 2
    s /= len(values)
    return np.sqrt(s)


def calc_p_value(mean, std, value):
    z_score = (value - mean) / std
    return scipy.stats.norm.sf(abs(z_score), loc=mean, scale=std)


def measure_lag(pred_labels, true_labels, injection_length, step_over, window_size, offset=0):
    # pred is for windows
    # true is for packets
    i = offset
    detected = 0
    missed = 0
    mean_lag = 0
    while i < len(true_labels):
        possible_windows = pred_labels[i: min(i + injection_length + step_over - window_size, len(pred_labels))]
        if max(possible_windows) == 1:
            detected += 1
            lag = -1
            for i in range(len(possible_windows)):
                if possible_windows[i] == 1:
                    lag = i
                    break
            mean_lag += lag
        else:
            missed += 1
        i += step_over + injection_length
    mean_lag /= detected
    return detected, missed, mean_lag


def count_outliers(svm_classification, window_size):
    counts = []
    for i in range(len(svm_classification) - window_size + 1):
        w = svm_classification[i: i + window_size]
        count = sum([1 if classification == -1 else 0 for classification in w])
        counts.append(count)

    return calc_MLE_mean(counts), cal_MLE_std(counts)


def create_val_for_DFA(group=None, val_data_path=None):
    if group is not None:
        with open(val_data_path, mode='rb') as val_f:
            pkts = pickle.load(val_f)
    else:
        pkts = data.load(data.datasets_path, "VAL")

    bins, binning_methods, names = get_DFA_params()

    processed = data.process(pkts, 'v3_2_abstract', None, None, False, registers=None, fill=False)
    registers = processed.columns[2:]

    # save the data set.
    val_path = val_base + '//DFA'
    if not os.path.exists(val_path):
        Path(val_path).mkdir(parents=True, exist_ok=True)

    for b in bins:
        for binning_method in binning_methods.keys():
            val_data = processed.copy()
            for col_name in processed.columns:
                if col_name in registers:
                    # load binner.
                    binner_path = binners_base + '//DFA//{}_{}_{}_{}'.format(
                        group, names[binning_method], b, col_name)
                    with open(binner_path, mode='rb') as binner_p:
                        col_binner = pickle.load(binner_p)
                    val_data[col_name] = col_binner.transform(
                        val_data[col_name].to_numpy().reshape(-1, 1))

            val_data = squeeze(val_data)

            with open(val_path + '//{}_X_{}_{}'.format(group, binning_method, b), mode='wb') as df_f:
                pickle.dump(val_data, df_f)

            transitions_labels, pkts_to_states = get_transitions_labels(pkts, [0] * len(pkts), val_data)

            with open(val_path + '//{}_dict_{}_{}'.format(group, binning_method, b), mode='wb') as dict_f:
                pickle.dump(pkts_to_states, dict_f)

            with open(val_path + '//{}_labels_{}_{}'.format(group, binning_method, b), mode='wb') as labels_f:
                pickle.dump(transitions_labels, labels_f)


def create_DFA_val_sets_single_plcs_split():
    for ip in data.active_ips:
        val_data_path = data.datasets_path + f'//single_plc//{ip}_val'
        create_val_for_DFA(group=f'single_plc_{ip}', val_data_path=val_data_path)


def create_DFA_val_sets_all_plcs_split():
    val_data_path = data.datasets_path + f'//all_plcs//val'
    create_val_for_DFA(group=f'all_plcs', val_data_path=val_data_path)


def create_DFA_val_sets_pearson_split():
    for group in pearson_groups:
        val_data_path = data.datasets_path + f'//pearson//{group}_val'
        create_val_for_DFA(group=group, val_data_path=val_data_path)


def create_DFA_val_sets_spearman_split():
    for group in pearson_groups:
        val_data_path = data.datasets_path + f'//spearman//{group}_val'
        create_val_for_DFA(group=group, val_data_path=val_data_path)


def create_DFA_val_sets_k_means_split():
    for group in pearson_groups:
        val_data_path = data.datasets_path + f'//k_means//{group}_val'
        create_val_for_DFA(group=group, val_data_path=val_data_path)


def reorder(df):
    kmeans = df.loc[df['binning'] == 'k_means']
    freq = df.loc[df['binning'] == 'equal_frequency']
    width = df.loc[df['binning'] == 'equal_width']
    sorted_df = df.sort_values(by=['algorithm', 'data version', 'binning', '# bins', 'percentage'],
                               ascending=[True, True, True, True, True])
    # return pd.concat([kmeans, freq, width], ignore_index=True)
    return sorted_df


def get_lstm_sheets_df():
    versions = ['v1_1', 'v2', 'v3_2']
    df = pd.DataFrame()
    for version in versions:
        file_name = xl_base + f'LSTM {version} sheets.xlsx'
        version_df = pd.read_excel(file_name, sheet_name='performance')
        df = pd.concat([df, reorder(version_df)], ignore_index=True)
    return df


def get_htm_sheets_df():
    versions = ['v1_1', 'v2', 'v3_2']
    df = pd.DataFrame()
    for version in versions:
        file_name = xl_base + f'HTM {version} sheets.xlsx'
        version_df = pd.read_excel(file_name, sheet_name='performance')
        df = pd.concat([df, reorder(version_df)], ignore_index=True)
    return df


def get_dfa_sheet_df():
    df = pd.read_excel(xl_base + 'DFA sheets.xlsx', sheet_name='performance')
    return df.loc[(df['window size'] == 200) & (df['# bins'] != 3)]


def create_performance_sheet():
    lstm, htm, dfa = get_lstm_sheets_df(), get_htm_sheets_df(), get_dfa_sheet_df()
    scores = pd.concat([lstm, htm, dfa], ignore_index=True)
    scores_best = pd.concat([lstm.loc[lstm['binning'] == 'equal_width'], htm.loc[htm['binning'] == 'equal_width'],
                             dfa.loc[dfa['binning'] == 'equal_frequency']], ignore_index=True)
    with pd.ExcelWriter(xl_base + 'full results.xlsx', mode="a", engine="openpyxl",
                        if_sheet_exists="overlay") as writer:
        scores.to_excel(writer, sheet_name='results')
        scores_best.to_excel(writer, sheet_name='best results')


def stat_test_algos():
    lstm = get_lstm_sheets_df()
    htm = get_htm_sheets_df()
    dfa = get_dfa_sheet_df()
    lstm_v1_1, lstm_v2, lstm_v3_2 = lstm.loc[lstm['data version'] == 'v1_1'], lstm.loc[lstm['data version'] == 'v2'], \
                                    lstm.loc[lstm['data version'] == 'v3_2']
    htm_v1_1, htm_v2, htm_v3_2 = htm.loc[htm['data version'] == 'v1_1'], htm.loc[htm['data version'] == 'v2'], htm.loc[
        htm['data version'] == 'v3_2']
    do_test(lstm_v1_1, dfa, 'lstm v1_1', 'dfa')
    do_test(lstm_v2, dfa, 'lstm v2', 'dfa')
    do_test(lstm_v3_2, dfa, 'lstm v3_2', 'dfa')

    do_test(htm_v1_1, dfa, 'htm v1_1', 'dfa')
    do_test(htm_v2, dfa, 'htm v2', 'dfa')
    do_test(htm_v3_2, dfa, 'htm v3_2', 'dfa')

    do_test(lstm_v1_1, htm_v1_1, 'lstm v11', 'htm v11')
    do_test(lstm_v2, htm_v2, 'lstm v2', 'htm v2')
    do_test(lstm_v3_2, htm_v3_2, 'lstm v3_2', 'htm v3_2')

    do_test(lstm_v3_2, lstm_v2, 'lstm v3_2', 'lstm v1_1')
    do_test(lstm_v3_2, lstm_v2, 'lstm v3_2', 'lstm v2')

    do_test(htm_v3_2, htm_v1_1, 'htm v3_2', 'htm v11')
    do_test(htm_v3_2, htm_v2, 'htm v3_2', 'htm v2')


def do_test(first, second, name1, name2):
    t_stat, p_value = stats.ttest_ind(first['f1'], second['f1'], alternative='greater')
    # Set the significance level (alpha)
    alpha = 0.05

    if p_value < alpha:
        print(f"There is a significant difference between {name1} and {name2} (p < 0.05).")
    else:
        print(f"There is no significant difference between {name1} and {name2} (p >= 0.05).")


def stat_test_feature_extraction():
    lstm = get_lstm_sheets_df()
    htm = get_htm_sheets_df()
    lstm_v1_1, lstm_v2, lstm_v3_2 = lstm.loc[lstm['data version'] == 'v1_1'], lstm.loc[lstm['data version'] == 'v2'], \
                                    lstm.loc[lstm['data version'] == 'v3_2']
    htm_v1_1, htm_v2, htm_v3_2 = htm.loc[htm['data version'] == 'v1_1'], htm.loc[htm['data version'] == 'v2'], htm.loc[
        htm['data version'] == 'v3_2']
    test1(lstm_v1_1['f1'], lstm_v2['f1'], lstm_v3_2['f1'])
    test1(htm_v1_1['f1'], htm_v2['f1'], htm_v3_2['f1'])


def test1(v1_1, v2, v3_2):
    # Perform a paired t-test between method3 and method1
    t_stat1, p_value1 = stats.ttest_ind(v3_2, v1_1, alternative='greater')
    t_stat2, p_value2 = stats.ttest_ind(v3_2, v2, alternative='greater')
    # Set the significance level (alpha)
    alpha = 0.05

    if p_value1 < alpha:
        print("There is a significant difference between Method 3 and Method 1 (p < 0.05).")
    else:
        print("There is no significant difference between Method 3 and Method 1 (p >= 0.05).")

    if p_value2 < alpha:
        print("There is a significant difference between Method 3 and Method 2 (p < 0.05).")
    else:
        print("There is no significant difference between Method 3 and Method 2 (p >= 0.05).")


def test_number_of_bins():
    lstm, htm, dfa = get_lstm_sheets_df(), get_htm_sheets_df(), get_dfa_sheet_df()
    dfa = dfa.loc[dfa['binning'] == 'equal_width']
    lstm_v3_2 = lstm.loc[(lstm['data version'] == 'v3_2') & (lstm['binning'] == 'equal_width')]
    htm_v3_2 = htm.loc[(htm['data version'] == 'v3_2') & (htm['binning'] == 'equal_width')]
    alpha = 0.05
    for n_bins in range(4, 8):
        res_lstm, res_htm, res_dfa = lstm_v3_2.loc[lstm_v3_2['# bins'] == n_bins]['f1'], \
                                     htm_v3_2.loc[htm_v3_2['# bins'] == n_bins]['f1'], \
                                     dfa.loc[dfa['# bins'] == n_bins]['f1']

        t_stat, p_value = stats.ttest_ind(res_lstm, res_dfa, alternative='greater')

        if p_value < alpha:
            print(f"There is a significant difference between LSTM and DFA with {n_bins} bins (p < 0.05).")
        else:
            print(f"There is no significant difference between LSTM and DFA with {n_bins} bins (p >= 0.05).")

        t_stat, p_value = stats.ttest_rel(res_htm, res_dfa)

        if p_value < alpha:
            print(f"There is a significant difference between HTM and DFA with {n_bins} bins (p < 0.05).")
        else:
            print(f"There is no significant difference between HTM and DFA with {n_bins} bins (p >= 0.05).")

        t_stat, p_value = stats.ttest_rel(res_lstm, res_htm)

        if p_value < alpha:
            print(f"There is a significant difference between LSTM and HTM with {n_bins} bins (p < 0.05).")
        else:
            print(f"There is no significant difference between LSTM and HTM with {n_bins} bins (p >= 0.05).")


def best_binning_for_algos():
    lstm, htm, dfa = get_lstm_sheets_df(), get_htm_sheets_df(), get_dfa_sheet_df()
    lstm_v3_2 = lstm.loc[lstm['data version'] == 'v3_2']
    htm_v3_2 = htm.loc[htm['data version'] == 'v3_2']
    alpha = 0.05
    res_lstm_ef, res_htm_ef, res_dfa_ef = lstm_v3_2.loc[lstm_v3_2['binning'] == 'equal_frequency']['f1'], \
                                          htm_v3_2.loc[htm_v3_2['binning'] == 'equal_frequency']['f1'], \
                                          dfa.loc[dfa['binning'] == 'equal_frequency']['f1']

    res_lstm_ew, res_htm_ew, res_dfa_ew = lstm_v3_2.loc[lstm_v3_2['binning'] == 'equal_width']['f1'], \
                                          htm_v3_2.loc[htm_v3_2['binning'] == 'equal_width']['f1'], \
                                          dfa.loc[dfa['binning'] == 'equal_width']['f1']

    res_lstm_k, res_htm_k, res_dfa_k = lstm_v3_2.loc[lstm_v3_2['binning'] == 'k_means']['f1'], \
                                       htm_v3_2.loc[htm_v3_2['binning'] == 'k_means']['f1'], \
                                       dfa.loc[dfa['binning'] == 'k_means']['f1']

    t_stat, p_value = stats.ttest_ind(res_lstm_ew, res_lstm_ef, alternative='greater')

    if p_value < alpha:
        print(f"There is a significant difference between LSTM: {'EW'}  {'EF'} binning (p < 0.05).")
    else:
        print(f"There is no significant difference (p >= 0.05).")

    t_stat, p_value = stats.ttest_ind(res_lstm_ew, res_lstm_k, alternative='greater')

    if p_value < alpha:
        print(f"There is a significant difference between LSTM: {'EW'}  {'K'} binning (p < 0.05).")
    else:
        print(f"There is no significant difference (p >= 0.05).")

    t_stat, p_value = stats.ttest_ind(res_htm_ew, res_htm_ef, alternative='greater')

    if p_value < alpha:
        print(f"There is a significant difference between HTM: {'EW'}  {'EF'} binning (p < 0.05).")
    else:
        print(f"There is no significant difference (p >= 0.05).")

    t_stat, p_value = stats.ttest_ind(res_htm_ew, res_htm_k, alternative='greater')

    if p_value < alpha:
        print(f"There is a significant difference between HTM: {'EW'}  {'K'} binning (p < 0.05).")
    else:
        print(f"There is no significant difference (p >= 0.05).")

    t_stat, p_value = stats.ttest_ind(res_dfa_ef, res_dfa_ew, alternative='greater')

    if p_value < alpha:
        print(f"There is a significant difference between dfa: {'EF'}  {'EW'} binning (p < 0.05).")
    else:
        print(f"There is no significant difference (p >= 0.05).")

    t_stat, p_value = stats.ttest_ind(res_dfa_ef, res_dfa_k, alternative='greater')

    if p_value < alpha:
        print(f"There is a significant difference between dfa: {'EF'}  {'K'} binning (p < 0.05).")
    else:
        print(f"There is no significant difference (p >= 0.05).")


def test_best_binning_method():
    lstm, htm, dfa = get_lstm_sheets_df(), get_htm_sheets_df(), get_dfa_sheet_df()
    lstm_v3_2 = lstm.loc[lstm['data version'] == 'v3_2']
    htm_v3_2 = htm.loc[htm['data version'] == 'v3_2']
    alpha = 0.05
    for binning_method in ['k_means', 'equal_frequency', 'equal_width']:
        res_lstm, res_htm, res_dfa = lstm_v3_2.loc[lstm_v3_2['binning'] == binning_method]['f1'], \
                                     htm_v3_2.loc[htm_v3_2['binning'] == binning_method]['f1'], \
                                     dfa.loc[dfa['binning'] == binning_method]['f1']

        t_stat, p_value = stats.ttest_ind(res_lstm, res_dfa, alternative='greater')

        if p_value < alpha:
            print(f"There is a significant difference between LSTM and DFA with {binning_method} binning (p < 0.05).")
        else:
            print(f"There is no significant difference between LSTM and DFA with {binning_method} binning (p >= 0.05).")

        t_stat, p_value = stats.ttest_ind(res_htm, res_dfa, alternative='greater')

        if p_value < alpha:
            print(f"There is a significant difference between HTM and DFA with {binning_method} binning (p < 0.05).")
        else:
            print(f"There is no significant difference between HTM and DFA with {binning_method} binning (p >= 0.05).")

        t_stat, p_value = stats.ttest_ind(res_lstm, res_htm, alternative='greater')

        if p_value < alpha:
            print(f"There is a significant difference between LSTM and HTM with {binning_method} binning (p < 0.05).")
        else:
            print(f"There is no significant difference between LSTM and HTM with {binning_method} binning (p >= 0.05).")


def test_for_each_percentage():
    lstm, htm, dfa = get_lstm_sheets_df(), get_htm_sheets_df(), get_dfa_sheet_df()
    lstm_v3_2 = lstm.loc[
        (lstm['data version'] == 'v3_2') & (lstm['binning'] == 'equal_width') & (lstm['percentage'] > 20)]
    htm_v3_2 = htm.loc[(htm['data version'] == 'v3_2') & (htm['binning'] == 'equal_width') & (htm['percentage'] > 20)]
    dfa = dfa.loc[(dfa['binning'] == 'equal_width') & (dfa['percentage'] > 20)]
    alpha = 0.05

    res_lstm, res_htm, res_dfa = lstm_v3_2['f1'], htm_v3_2['f1'], dfa['f1']

    t_stat, p_value = stats.ttest_ind(res_lstm, res_dfa, alternative='greater')

    if p_value < alpha:
        print(f"There is a significant difference between LSTM and DFA with percentage (p < 0.05).")
    else:
        print(f"There is no significant difference between LSTM and DFA with percentage (p >= 0.05).")

    t_stat, p_value = stats.ttest_ind(res_htm, res_dfa, alternative='greater')

    if p_value < alpha:
        print(f"There is a significant difference between HTM and DFA with percentage (p < 0.05).")
    else:
        print(f"There is no significant difference between HTM and DFA with percentage (p >= 0.05).")

    t_stat, p_value = stats.ttest_ind(res_lstm, res_htm, alternative='greater')

    if p_value < alpha:
        print(f"There is a significant difference between LSTM and HTM with percentage (p < 0.05).")
    else:
        print(f"There is no significant difference between LSTM and HTM with percentage (p >= 0.05).")


def plot_anlis_htm(data_version, binning, bins):
    percentages = [40, 50, 60, 70, 80]
    name = 'test_{}_{}_{}'.format(data_version, binning, bins)
    # anlis_path = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\anomaly likelihoods\\'
    anlis_path = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\anlis hws=2k\\'
    avg_anlis = None
    pass_threshold = []
    for p in percentages:
        full_name = name + '_{}'.format(p)
        anli_path = anlis_path + full_name
        with open(anli_path, mode='rb') as f:
            anlis = pickle.load(f, encoding='latin1')
            if avg_anlis:
                avg_anlis = [avg_anlis[i] + anlis[i] for i in range(len(anlis))]
            else:
                avg_anlis = anlis
            """plt.plot([i + 1 for i in range(len(anlis))], [1 if anli > 0.99 else 0 for anli in anlis], label=p)
            plt.xlabel('Index')
            plt.ylabel('ANLI Values')
            plt.title('ANLI List Plot')
            plt.grid(True)
            plt.show()"""
            print(p, len([x for x in anlis if x > 0.99]))
            print([i for i in range(len(anlis)) if anlis[i] > 0.99])
            pass_threshold.append(len([x for x in anlis if x > 0.99]))
    return pass_threshold


def process_htm_log():
    with open('C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\HTM\\test HTM-OCSVM.txt', 'r') as file:
        lines = file.readlines()
    htm_df = get_htm_sheets_df()
    htm_df['tpr'] = 0
    htm_df['fpr'] = 0
    htm_df['tnr'] = 0
    htm_df['fnr'] = 0

    tprs, fprs, tnrs, fnrs = [], [], [], []

    num_lines = len(lines)
    for i in range(0, num_lines, 8):
        result = lines[i: i + 8]
        needed_metrics_line = result[-1]

        tp, fp, tn, fn = get_metrics_from_line(needed_metrics_line)
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        fpr = fp / (fp + tn)
        fnr = fn / (tn + tp)

        tprs.append(tpr)
        tnrs.append(tnr)
        fprs.append(fpr)
        fnrs.append(fnr)

    htm_df['tpr'] = tprs
    htm_df['fpr'] = fprs
    htm_df['tnr'] = tnrs
    htm_df['fnr'] = fnrs
    with pd.ExcelWriter('C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\HTM\\htm analysis.xlsx', mode="a",
                        engine="openpyxl",
                        if_sheet_exists="overlay") as writer:
        htm_df.to_excel(writer, sheet_name='full results')


def get_metrics_from_line(line):
    parts = line.split(',')
    tp, fp, fn, tn = None, None, None, None

    # Iterate through the parts to find the values
    for part in parts:
        key, value = part.split(':')
        if key.strip() == "tp":
            tp = float(value.strip())
        elif key.strip() == "fp":
            fp = float(value.strip())
        elif key.strip() == "fn":
            fn = float(value.strip())
        elif key.strip() == "tn":
            tn = float(value.strip())
    return tp, fp, tn, fn


def lag_in_HTM_anli(data_version, binning, bins):
    percentages = [40, 50, 60, 70, 80]
    name = 'test_{}_{}_{}'.format(data_version, binning, bins)
    anlis_path = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\anomaly likelihoods\\'

    for p in percentages:
        full_name = name + '_{}'.format(p)
        anli_path = anlis_path + full_name

        with open(anli_path, mode='rb') as f:
            anlis = pickle.load(f, encoding='latin1')

        below_thresh = 0

        for i in range(0, len(anlis), 3000):
            found = False
            j = 0
            while j < 100 and not found:
                if anlis[i + j] < 0.99:
                    below_thresh += 1
                j += 1
        # delay /= 10
        print(below_thresh)


def examine_datasets_for_many_PLCs():
    candidates = ['MB_TCP', 'modbus12']
    # show: how many PLCs involved.
    # show: how many HTMs involved.
    # show: number of packets to each PLCs.
    mb_tcp_p = data.datasets_path + f'\\{candidates[0]}'
    modbus_12_p = data.datasets_path + f'\\{candidates[1]}'

    # with open(mb_tcp_p, mode='rb') as df_f:
    # mb_tcp = pickle.load(df_f)

    with open(modbus_12_p, mode='rb') as df_f:
        modbus_12 = pickle.load(df_f)

    # print('examine mb_tcp')
    # examine_raw_df(mb_tcp)

    print('examine mobdbus_12')
    examine_raw_df(modbus_12)


def examine_raw_df(df):
    responses = df.loc[df['src_port'] == data.plc_port]
    PLCs = responses['src_ip'].unique()
    print(f'there are {len(PLCs)} plcs.')

    for plc in PLCs:
        queries = df.loc[df['dst_ip'] == plc]
        responses = df.loc[df['src_ip'] == plc]
        print(f'PLC:{plc}, # queries: {len(queries)}, # responses: {len(responses)}')


def analyze_htm_log_for_scores():
    log = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\test HTM-OCSVM.txt'
    with open(log, 'r') as file:
        log_entries = file.readlines()

    # Initialize lists to store extracted data
    percentages = []
    data_versions = []
    binning_methods = []
    bins = []
    windows = []
    std_counts = []
    likelihood_thresholds = []
    aucs = []
    prcs = []
    f1s = []
    precisions = []
    recalls = []

    # Regular expressions to match the patterns and extract data
    percent_pattern = re.compile(r'%:\s(\d+)')
    data_version_pattern = re.compile(r'data version:\s(\S+)')
    binning_method_pattern = re.compile(r'binning method:(equal_frequency|equal_width|k_means)')
    bins_pattern = re.compile(r'bins:(\d+)')
    window_pattern = re.compile(r'window:(\d+)')
    std_count_pattern = re.compile(r'std count:(\d+)')
    auc_pattern = re.compile(r'auc:([\d.]+)')
    prc_pattern = re.compile(r'prc:([\d.]+)')
    f1_pattern = re.compile(r'f1:([\d.]+)')
    precision_pattern = re.compile(r'precision:([\d.]+)')
    recall_pattern = re.compile(r'recall:([\d.]+)')
    likelihood_threshold_pattern = re.compile(r'likelihood threshold:\s([\d.]+)')

    # Iterate through log entries and extract the data
    for i in range(0, len(log_entries), 8):
        entry = log_entries[i: i + 8]
        percent_match = percent_pattern.search(entry[0])
        data_version_match = data_version_pattern.search(entry[2])
        binning_method_match = binning_method_pattern.search(entry[2])
        bins_match = bins_pattern.search(entry[2])
        window_match = window_pattern.search(entry[2])
        std_count_match = std_count_pattern.search(entry[2])
        likelihood_threshold_match = likelihood_threshold_pattern.search(entry[2])

        auc_match = auc_pattern.search(entry[7])
        prc_match = prc_pattern.search(entry[7])
        f1_match = f1_pattern.search(entry[7])
        precision_match = precision_pattern.search(entry[7])
        recall_match = recall_pattern.search(entry[7])

        if data_version_match.group(1)[:-1] != 'v3_2':
            continue

        if percent_match:
            percentages.append(percent_match.group(1))
        if data_version_match:
            data_versions.append(data_version_match.group(1)[:-1])
        if binning_method_match:
            binning_methods.append(binning_method_match.group(1))
        if bins_match:
            bins.append(bins_match.group(1))
        if window_match:
            windows.append(window_match.group(1))
        if std_count_match:
            std_counts.append(std_count_match.group(1))
        if likelihood_threshold_match:
            likelihood_thresholds.append(likelihood_threshold_match.group(1))

        if auc_match:
            aucs.append(auc_match.group(1))
        if prc_match:
            prcs.append(prc_match.group(1))
        if f1_match:
            f1s.append(f1_match.group(1))
        if precision_match:
            precisions.append(precision_match.group(1))
        if recall_match:
            recalls.append(recall_match.group(1))

    # Create a DataFrame from the extracted data
    data = {
        k: ['-'] * len(percentages) for k in excel_cols
    }

    data['percentage'] = percentages
    data['injection length'] = 100
    data['data version'] = data_versions
    data['binning'] = binning_methods
    data['# bins'] = bins
    data['window size'] = windows
    data['# std count'] = std_counts
    data['likelihood_threshold'] = likelihood_thresholds
    data['auc'] = aucs
    data['prc'] = prcs
    data['f1'] = f1s
    data['precision'] = precisions
    data['recall'] = recalls
    data['algorithm'] = 'HTM'

    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file
    df.to_excel('HTM v3_2 sheets compl.xlsx', index=False, engine='openpyxl')


def test_scalability():
    # 0. for each percentage of the data to use:
    # 1. create the raw train set
    # 2. train the LSTMs  and OCSVMs (each LSTM records in CV and fitting time, and OCSVM too)
    # 3. test the method, record inference times
    # 4. train the DFAs (each DFA records the train time)
    # 5. test DFAs, record inference times.
    LSTM_train_config = ...
    OCSVM_train_config = ...
    injection_config = ...

    for dataset_percentage in [20, 40, 60, 80, 100]:
        modbus_train_set = data.load(data.datasets_path, "TRAIN")
        data_to_use = int((dataset_percentage / 100) * len(modbus_train_set))
        modbus_train_set = modbus_train_set[:data_to_use]

        train_LSTM(LSTM_train_config, train_df=modbus_train_set, group_info=None)
        train_OCSVM(OCSVM_train_config_file_path=OCSVM_train_config, group_info=None)
        test_LSTM_based_OCSVM(LSTM_train_config, OCSVM_train_config, injection_config)

        train_automaton(group=None, train_data=modbus_train_set)
        test_DFA(injection_config=injection_config)


def add_entry_to_df(result_dict, df_cols, excel_path, sheet_name):
    res_df = pd.DataFrame.from_dict(data={'0': result_dict},
                                    columns=df_cols,
                                    orient='index')

    row = 0

    with pd.ExcelWriter(excel_path, mode="a", engine="openpyxl",
                        if_sheet_exists="overlay") as writer:
        if sheet_name in writer.sheets.keys():
            row = writer.sheets[sheet_name].max_row
        h = True
        if row > 0:
            h = False
        res_df.to_excel(excel_writer=writer, sheet_name=sheet_name,
                        startrow=row, header=h, index=False)


if __name__ == '__main__':
    # create_performance_sheet()
    # test_number_of_bins()
    # test_best_binning_method()
    # best_binning_for_algos()
    # test_for_each_percentage()
    # stat_test_algos()
    # examine_datasets_for_many_PLCs()
    print('hello world')
    test_LSTM_OCSVM_all_splits_all_injections("lstm_config", "ocsvm_config", "injection_config")
    # test_DFA_all_splits_all_injections("injection_config")
