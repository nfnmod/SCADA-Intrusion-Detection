import csv
import itertools
import os
import pickle
import time
from pathlib import Path

import keras.models
import numpy as np
import pandas as pd
import scipy
import tensorflow
import yaml
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score, r2_score, auc, precision_score, \
    recall_score, mean_squared_error
from sklearn.svm import OneClassSVM
from scipy.spatial import distance

import data
import models
import models.TIRP as TIRP
from data import squeeze, find_frequent_transitions_sequences
from data.injections import inject_to_raw_data

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
default_supp = 15
KL_OCSVM_datasets = models.SCADA_base + '\\KL_OCSVM datasets'
KL_OCSVM_base = models.SCADA_base + '\\KL_OCSVM'
KL_test_sets_base = "C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\test sets\\KL\\KL out"

excel_cols_old = {'HTM type', 'LSTM type', 'mix', 'data version', 'binning', '# bins', 'nu', 'kernel', '# estimators',
                  'criterion', 'max features',
                  'ON bits', 'SDR size', 'numOfActiveColumnsPerInhArea', 'potential Pct', 'synPermConnected',
                  'synPermActiveInc', 'synPermInactiveDec', 'boostStrength', 'cellsPerColumn', 'newSynapseCount',
                  'initialPerm', 'permanenceInc', 'permanenceDec', 'maxSynapsesPerSegment', 'maxSegmentsPerCell',
                  'minThreshold', 'activationThreshold', 'window size', 'KL epsilon', 'minimal VS', 'max gap',
                  'injection length', 'step over', 'percentage', 'precision', 'recall', 'auc', 'f1', 'prc'}
excel_cols = ['algorithm', 'data version', 'binning', '# bins', '# std', '# std count', 'p_value', 'distance metric', 'window size',
              'ON bits', 'SDR size', 'numOfActiveColumnsPerInhArea', 'potential Pct', 'synPermConnected',
              'synPermActiveInc', 'synPermInactiveDec', 'boostStrength', 'cellsPerColumn', 'newSynapseCount',
              'initialPerm', 'permanenceInc', 'permanenceDec', 'maxSynapsesPerSegment', 'maxSegmentsPerCell',
              'minThreshold', 'activationThreshold', 'window size', 'KL epsilon', 'minimal VS', 'max gap',
              'injection length', 'percentage', 'precision', 'recall', 'auc', 'f1', 'prc']
RF_cols = {'data version', 'binning', '# bins', '# estimators', 'criterion', 'max features',
           'injection length', 'step over', 'percentage', 'precision', 'recall', 'auc', 'f1', 'prc'}
OCSVM_cols = {'data version', 'binning', '# bins', 'nu', 'kernel',
              'injection length', 'step over', 'percentage', 'precision', 'recall', 'auc', 'f1', 'prc'}

DFA_cols = {'binning', '# bins', 'injection length', 'step over', 'percentage', 'precision',
            'recall', 'auc', 'f1', 'prc'}

LSTM_detection_cols = ['data version', 'binning', '# bins', '# std', '# std count', 'p_value', 'distance metric', 'injection length', 'step over',
                       'percentage',
                       'precision',
                       'recall', 'auc', 'f1', 'prc']

KL_based_RF_cols = {'binning', '# bins', 'window size', 'KL epsilon', 'minimal VS', 'max gap',
                    'injection length', 'step over', 'percentage', 'precision', 'recall', 'auc',
                    'f1', 'prc'}

window_sizes = [400, 600, 800, 1000]
p_values = [0.01, 0.03, 0.05]

nums_std = [0, 1, 2, 3]
num_stds_count = [0, 1, 2, 3]

best_cols = DFA_cols.copy()
lim = 0.2

xl_path = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\excel\\classifiers comprison.xlsx'

xl_labels_path = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\excel\\labels.xlsx'

test_LSTM_RF_OCSVM_log = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\log files\\test LSTM-RF-OCSVM.txt'
test_DFA_log = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\log files\\test DFA.txt'
test_KL_RF_log = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\log files\\test KL-RF.txt'
test_LSTM_STD_log = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\log files\\test LSTM_STD.txt'

LSTM_validation = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\datasets\\validation\\LSTM\\'


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


def train_OCSVM(OCSVM_train_config_file_path, many=False):
    with open(OCSVM_train_config_file_path, mode='r') as train_config:
        models_folders, data_folders, binning_dict, params = get_models_folders_data_folders(train_config)
        zipped = zip(models_folders, data_folders)
        for folder_pair in zipped:
            models_folder = folder_pair[0]
            data_folder = folder_pair[1]
            if not many:
                models.models.make_classifier(models_folder=models_folder, data_folder=data_folder,
                                              params=params, OCSVM_only=True)
            else:
                for group in params['groups']:
                    models_folder = group + '_' + models_folder
                    data_folder = group + '_' + data_folder
                    models.models.make_classifier(models_folder=models_folder, data_folder=data_folder,
                                                  params=params, OCSVM_only=True)


# single plc or groups of plcs.
def train_automaton(group=None):
    """
    :param group: group name, if none then we do it for a single plc.
                  otherwise, it can be used to train a PLC for each "real" group or for each PLC separately
                  it depends on the partitioning of the PLCs into group.
    :return:
    """
    if group is not None:
        pkts = data.load(data.datasets_path, "MODBUS_TCP_TRAIN_{}".format(group))
    else:
        pkts = data.load(data.datasets_path, "MODBUS_TCP_TRAIN")

    bins, binning_methods, names = get_DFA_params()

    processed = data.process(pkts, 'v3_2_abstract', None, None, False)
    registers = processed.columns[2:]

    binner_path = binners_base + '//DFA'

    if not os.path.exists(binner_path):
        Path(binner_path).mkdir(parents=True, exist_ok=True)

    for b in bins:
        for binning_method in binning_methods.keys():
            train_data = processed.copy()
            for col_name in processed.columns:
                if col_name in registers:
                    binner = binning_methods[binning_method]
                    binner_full_path = 'DFA//{}_{}'.format(names[binning_method], b)
                    if not os.path.exists(binner_full_path):
                        Path(binners_base + '//' + binner_full_path).mkdir(parents=True, exist_ok=True)
                    train_data[col_name] = binner(train_data, col_name, b, binner_full_path)

            with open(DFA_log, mode='a') as log:
                log.write('Creating DFA\n')

            dfa_input = squeeze(train_data)

            start = time.time()
            automaton = models.automaton.make_automaton(registers, dfa_input)
            end = time.time()

            with open(DFA_log, mode='a') as log:
                log.write('Done, time elapsed:{}\n'.format(end - start))

            with open(data.automaton_path + '\\{}_{}'.format(names[binning_method], b), mode='wb') as dfa_path:
                pickle.dump(automaton, dfa_path)


def train_automatons(groups):
    for group in groups:
        train_automaton(group)


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


def make_input_for_KL(TIRP_config_file_path):
    pkt_df = data.load(data.datasets_path, "TRAIN")
    IP = data.plc

    # consider only response packets from the PLC.
    plc_df = pkt_df.loc[(pkt_df['src_ip'] == IP)]
    with open(TIRP_config_file_path, mode='r') as train_config:
        params = yaml.load(train_config, Loader=yaml.FullLoader)

    discovery_params = params['TIRP_discovery_params']
    binning = discovery_params['binning']
    binning_methods = {'KMeans': TIRP.k_means_binning, 'EqualFreq': TIRP.equal_frequency_discretization,
                       'EqualWidth': TIRP.equal_width_discretization}
    number_of_bins = discovery_params['number_of_bins']
    windows = window_sizes
    bins_window_options = itertools.product(number_of_bins, windows)
    options = itertools.product(binning, bins_window_options)

    for option in options:
        b = binning_methods[option[0]]
        k = option[1][0]
        w = option[1][1]
        TIRP.make_input(plc_df, b, k, w, regs_to_use=data.most_used, consider_last=True)


def filter_TIRPs(KL_config_file_path):
    """
    we discover all the TIRPs with a very low support threshold and then filter out the ones having higher support
    to avoid running KarmaLego many times.
    :param KL_config_file_path: KL params
    :return:
    """
    # 1. get to the file containing the mined TIRPs with very low support.
    binning, bins, windows, epsilons, max_gaps, min_horizontal_supps = get_KL_params(KL_config_file_path)

    binning_times_bins = itertools.product(binning, bins)
    parent_folders = itertools.product(windows, binning_times_bins)

    for window_binning_bins in parent_folders:
        # parameters of the events making.
        window = window_binning_bins[0]
        binning = window_binning_bins[1][0]
        bins = window_binning_bins[1][1]

        windows_folders_folder = KL_base + "{}_bins_{}_window_{}_out".format(binning, bins, window)

        max_gaps_times_supports = itertools.product(max_gaps, min_horizontal_supps)
        KL_hyperparams = itertools.product(epsilons, max_gaps_times_supports)

        # parameters of KL.
        for eps_gap_supp in KL_hyperparams:
            epsilon = eps_gap_supp[0]
            max_gap = eps_gap_supp[1][0]
            horizontal_supp = eps_gap_supp[1][1]

            # we will save the filtered TIRPs here.
            dest_path_suffix = "\\eps_{0}_minHS%_{1}_maxGap_{2}".format(epsilon, horizontal_supp, max_gap)
            # path to read the whole set of TIRPs from.
            src_path_suffix = "\\eps_{0}_minHS_{1}_maxGap_{2}".format(epsilon, default_supp, max_gap)

            windows_outputs_destination_folder_path = windows_folders_folder + dest_path_suffix
            windows_outputs_src_folder_path = windows_folders_folder + src_path_suffix

            if not os.path.exists(windows_outputs_destination_folder_path):
                Path(windows_outputs_destination_folder_path).mkdir(parents=True)

            # call helper function to finish.
            filter_and_write(windows_outputs_src_folder_path, windows_outputs_destination_folder_path, horizontal_supp)


# switch to filtering by a percentile!
def filter_and_write(windows_outputs_src_folder_path, windows_outputs_destination_folder_path, horizontal_supp):
    # iterate over the TIRPs in the windows files in the src folder.
    # read every TIRP line in the file and check for the HS. if it's high enough write it to the destination file.
    for TIRPs_in_window in os.listdir(windows_outputs_src_folder_path):
        window_file = windows_outputs_src_folder_path + '\\' + TIRPs_in_window
        dst_window_file = windows_outputs_destination_folder_path + '\\' + TIRPs_in_window
        numbers_of_instances = []
        for TIRP_line in window_file:
            tirp = TIRP.parse_line(TIRP_line)
            numbers_of_instances.append(TIRP.get_number_of_instances(tirp.instances))

        percentile_hs = np.percentile(numbers_of_instances, horizontal_supp)

        for TIRP_line in window_file:
            # parse into TIRP object
            tirp = TIRP.parse_line(TIRP_line)
            # filter by horizontal support.
            if TIRP.get_number_of_instances(tirp.instances) >= percentile_hs:
                if not os.path.exists(dst_window_file):
                    Path(dst_window_file).mkdir(parents=True)
                # write TIRP.
                with open(dst_window_file, mode='a') as dst_p:
                    dst_p.write(TIRP_line + '\n')


def train_LSTMs_from_KL(KL_config_file_path):
    # go over all KL configurations:
    binning, bins, windows, epsilons, max_gaps, min_horizontal_supps = get_KL_params(KL_config_file_path)
    look_back = [20]

    for binning_method in binning:
        for number_of_bins in bins:
            for window_size in windows:

                windows_folders_folder = KL_base + "{}_bins_{}_window_{}_out".format(binning_method, number_of_bins,
                                                                                     window_size)

                for epsilon in epsilons:
                    for max_gap in max_gaps:
                        for min_horizontal_supp_percentile in min_horizontal_supps:

                            path_suffix = "\\eps_{0}_minHS%_{1}_maxGap_{2}".format(epsilon,
                                                                                   min_horizontal_supp_percentile,
                                                                                   max_gap)
                            windows_outputs_folder_path = windows_folders_folder + path_suffix
                            TIRP_path = TIRPs_base + '\\{}_{}_{}_{}_{}_{}'.format(binning, bins, window_size, epsilon,
                                                                                  min_horizontal_supp_percentile,
                                                                                  max_gap)

                            # for each configuration: call parse_output.
                            TIRP_df = TIRP.output.parse_output(windows_outputs_folder_path, train=True,
                                                               tirps_path=TIRP_path)
                            # train LSTM.
                            for series_len in look_back:
                                model_name = '{}_{}_{}_{}_{}_{}'.format(binning_method, number_of_bins, window_size,
                                                                        epsilon, max_gap,
                                                                        min_horizontal_supp_percentile)
                                train_data_path = KL_base + 'KL_LSTM'
                                models_path = data.modeles_path + '\\KL_LSTM'
                                with open(KL_LSTM_log, mode='a') as log:
                                    log.write('Training: {}_{}_{}_{}_{}_{}'.format(binning_method, number_of_bins,
                                                                                   window_size, epsilon, max_gap,
                                                                                   min_horizontal_supp_percentile))

                                start = time.time()
                                models.simple_LSTM(TIRP_df, series_len, 42, model_name, train=1,
                                                   models_path=models_path, data_path=train_data_path)
                                end = time.time()
                                with open(KL_LSTM_log, mode='a') as log:
                                    log.write("trained, time elapsed: {}".format(end - start))


##########################################################################################
# IRRELEVANT FOR NOW.
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
    min_horizontal_supps = KL_params['MinVerSups']
    look_back = [20, 30]
    OCSVM_params = params['OCSVM']
    nus = OCSVM_params['nu']
    kernels = OCSVM_params['kernel']

    for binning_method in binning:
        for number_of_bins in bins:
            for window_size in windows:
                for epsilon in epsilons:
                    for max_gap in max_gaps:
                        for min_horizontal_supp in min_horizontal_supps:
                            # get LSTM and LSTM data.
                            for series_len in look_back:
                                model_name = '{}_{}_{}_{}_{}_{}_{}'.format(binning_method, number_of_bins, window_size,
                                                                           epsilon, max_gap, min_horizontal_supp,
                                                                           series_len)
                                train_data_folder_path = KL_base
                                models_path = data.modeles_path + '\\KL_LSTM'
                                LSTM = keras.models.load_model(models_path + '\\' + model_name)

                                with open(train_data_folder_path + '\\X_train_{}'.format(model_name),
                                          mode='rb') as data_p:
                                    X_train = pickle.load(data_p)

                                predictions = LSTM.predict(X_train)
                                predictions_path = KL_OCSVM_datasets + '\\X_train_{}_{}_{}_{}_{}_{}_{}'.format(
                                    binning_method, number_of_bins, window_size, epsilon, max_gap, min_horizontal_supp,
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
                                                                                                       min_horizontal_supp,
                                                                                                       series_len,
                                                                                                       kernel, nu))
                                        ocsvm = OneClassSVM(kernel=kernel, nu=nu)
                                        start = time.time()
                                        ocsvm.fit(predictions)
                                        end = time.time()
                                        with open(KL_based_OCSVM_log, mode='a') as log:
                                            log.write('Trained, time elapsed: {}'.format(end - start))
                                        # save model.
                                        model_path = KL_OCSVM_base + '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(binning_method,
                                                                                                         number_of_bins,
                                                                                                         window_size,
                                                                                                         epsilon,
                                                                                                         max_gap,
                                                                                                         min_horizontal_supp,
                                                                                                         series_len,
                                                                                                         kernel, nu)
                                        keras.models.save_model(ocsvm, model_path)


##########################################################################################
# process the raw data using some method without binning but with scaling. Then split the data, convert to csv and save.
# The csv file is formatted by the requirements of HTM.
def create_data_for_HTM(HTM_input_creation_config, many=False, g_ids=None):
    """
    open file, for each data version:
                    go over all bins, methods combinations and apply data processing.
                    save the train and test sets.
    """
    if not many:
        pkt_df = data.load(data.datasets_path, "modbus")
        helper(HTM_input_creation_config, pkt_df)
    else:
        for g in g_ids:
            with open(group_df_base + '_' + g, mode='rb') as group_df_path:
                g_df = pickle.load(group_df_path)
                helper(HTM_input_creation_config, g_df, g)


# run with x_val to create validation sets.
def helper(HTM_input_creation_config, pkt_df, g='', df_type='train'):
    with open(HTM_input_creation_config, mode='r') as input_config:
        params = yaml.load(input_config, Loader=yaml.FullLoader)
        versions_dicts = params['processing_config']
        for version_dict in versions_dicts:
            data_version = version_dict['name']
            use = version_dict['use']
            if not use:
                pass
            else:
                # X_train will be used to train the HTM network.
                X_train = data.process(pkt_df, data_version, None, None)
                # 1. write column names.
                # 2. write columns data types.
                # 3. write df to csv without the columns names.
                folder = HTM_base + '\\datasets\\' + '{}'.format(data_version)

                train_path_str = folder + '\\' + "X_" + df_type + "_" + g + '_' + data_version + ".csv"
                train_path = Path(train_path_str)

                train_path.parent.mkdir(parents=True, exist_ok=True)

                with open(train_path_str, 'w', newline='') as train_file:
                    train_writer = csv.writer(train_file)
                    # write the field names.
                    train_cols = list(X_train.columns)
                    train_writer.writerow(train_cols)
                    # write the field types.
                    train_cols_types = ['float'] * len(train_cols)
                    train_writer.writerow(train_cols_types)
                    # use no flags.
                    train_writer.writerow([])
                X_train.to_csv(path_or_buf=train_path, index=False, header=False, mode='a')


"""
The following function create the test files for the various classifiers.
Use configuration file to create them.
"""


def create_test_files_HTM(raw_test_data_df, data_versions_config, injection_config, group_id=''):
    """
    grid over injection params, for each combination : inject anomalies and then process the dataset using all methods.
    when doing this for many PLCs, need to do this for every group separately.
    """
    test_data = data.load(test_sets_base_folder, raw_test_data_df)
    lim = 0.2  # don't allow more than 20 percent of malicious packets in the data set.
    with open(injection_config, mode='r') as anomalies_config:
        injection_params = yaml.load(anomalies_config, Loader=yaml.FullLoader)

    injection_lengths = injection_params['InjectionLength']
    step_overs = injection_params['StepOver']
    percentages = injection_params['percentage']
    epsilons = injection_params['Epsilon']
    # first ,inject anomalies. and create the test set.
    for injection_length in injection_lengths:
        for step_over in step_overs:
            anomaly_percentage = injection_length / (injection_length + step_over)
            if anomaly_percentage > lim:
                pass
            else:
                for percentage in percentages:
                    for epsilon in epsilons:
                        anomalous_data, labels = inject_to_raw_data(test_data, injection_length, step_over,
                                                                    percentage,
                                                                    epsilon)
                        #  process the data using the different versions
                        with open(data_versions_config, mode='r') as processing_config:
                            config = yaml.load(processing_config, Loader=yaml.FullLoader)

                        data_versions = config['processing_config']

                        for data_version in data_versions:
                            to_use = data_version['use']
                            if not to_use:
                                pass
                            else:
                                name = data_version['name']

                                # same thing but for HTM. no need to save y_test because HTM predicts anomaly scores.
                                # no binning for HTM, only scaling.
                                test_df = data.process(anomalous_data, name, None, None)
                                suffix = '_{}_{}_{}_{}'.format(
                                    name, injection_length,
                                    step_over, percentage)
                                if group_id != '':
                                    suffix = group_id + suffix
                                p_x_test_HTM = test_sets_base_folder + '\\HTM\\X_test_' + suffix + '.csv'
                                p_labels_HTM = test_sets_base_folder + '\\HTM\\labels_' + suffix

                                if not os.path.exists(test_sets_base_folder + '\\HTM'):
                                    Path(test_sets_base_folder + '\\HTM').mkdir(parents=True,
                                                                                exist_ok=True)

                                with open(p_x_test_HTM, mode='w', newline='') as test_file:
                                    writer = csv.writer(test_file)
                                    test_cols = list(test_df.columns)
                                    # write columns names
                                    writer.writerow(test_cols)
                                    # write columns types
                                    columns_types = ['float'] * len(test_cols)
                                    # no flags
                                    writer.writerow(columns_types)
                                    writer.writerow([])
                                test_df.to_csv(path_or_buf=p_x_test_HTM, index=False, header=False,
                                               mode='a')

                                with open(p_labels_HTM, mode='wb') as labels_path:
                                    pickle.dump(labels, labels_path)


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


def create_test_files_DFA(injection_config, group=''):
    """
     just grid over all the injection parameters and binning params.
    """
    folders_names = {'k_means': 'KMeans', 'equal_frequency': 'EqualFreq',
                     'equal_width': 'EqualWidth'}
    n_bins, binners, names = get_DFA_params()
    injection_lengths, step_overs, percentages, epsilons = get_injection_params(injection_config)

    for injection_length in injection_lengths:
        for step_over in step_overs:
            anomaly_percentage = injection_length / (injection_length + step_over)
            if anomaly_percentage > lim:
                pass
            else:
                for percentage in percentages:
                    for epsilon in epsilons:
                        parent = test_sets_base_folder + '\\raw'
                        data_path = parent + '\\data_{}_{}_{}_{}'.format(injection_length, step_over, percentage,
                                                                         epsilon)
                        labels_path = parent + '\\labels_{}_{}_{}_{}'.format(injection_length, step_over, percentage,
                                                                             epsilon)

                        with open(data_path, mode='rb') as d_p:
                            anomalous_data = pickle.load(d_p)
                        with open(labels_path, mode='rb') as l_p:
                            labels = pickle.load(l_p)

                        test_df = data.process(anomalous_data, 'v3_2_abstract', None, None, False)
                        for binner in binners:
                            for bins in n_bins:
                                binned_test_df = test_df.copy()
                                for col_name in test_df.columns:
                                    if 'time' not in col_name:
                                        # load binner.
                                        binner_path = binners_base + '//DFA//{}_{}_{}'.format(
                                            folders_names[names[binner]], bins, col_name)
                                        with open(binner_path, mode='rb') as binner_p:
                                            col_binner = pickle.load(binner_p)
                                        binned_test_df[col_name] = col_binner.transform(
                                            binned_test_df[col_name].to_numpy().reshape(-1, 1))

                                dfa_in = squeeze(binned_test_df)
                                # we need to know which transitions include anomalies to create the test labels.
                                # iterate over anomalous_data, if a packet is anomalous, mark the transition from its' corresponding
                                # state as anomalous.

                                transitions_labels, pkts_to_states = get_transitions_labels(anomalous_data, labels,
                                                                                            dfa_in)

                                if group != '':
                                    middle = '\\DFA_{}'.format(group)
                                else:
                                    middle = '\\DFA'

                                p_x_test = test_sets_base_folder + middle + '\\X_test_{}_{}{}_{}_{}'.format(
                                    names[binner], bins, injection_length,
                                    step_over,
                                    percentage)

                                p_labels = test_sets_base_folder + middle + '\\labels_{}_{}_{}_{}_{}'.format(
                                    names[binner], bins, injection_length,
                                    step_over,
                                    percentage)

                                p_pkt_dict = test_sets_base_folder + middle + '\\labels_{}_{}_{}_{}_{}'.format(
                                    names[binner], bins, injection_length,
                                    step_over,
                                    percentage)

                                if not os.path.exists(test_sets_base_folder + middle):
                                    Path(test_sets_base_folder + middle).mkdir(parents=True, exist_ok=True)

                                with open(p_x_test, mode='wb') as test_path:
                                    pickle.dump(dfa_in, test_path)
                                with open(p_labels, mode='wb') as p_labels:
                                    pickle.dump(transitions_labels, p_labels)
                                with open(p_pkt_dict, mode='wb') as p_pkts_dict:
                                    pickle.dump(pkts_to_states, p_pkts_dict)


def create_test_input_TIRP_files_for_KL(raw_test_data_df, injection_config, input_creation_config):
    """
    make test data sets for KL.
    for each input creation option: create TIRP with all possible injection options.
    """
    name_2_func = {'EqualFreq': TIRP.equal_frequency_discretization, 'EqualWidth': TIRP.equal_width_discretization,
                   'KMeans': TIRP.k_means_binning}
    func_2_name = {TIRP.k_means_binning: 'kmeans', TIRP.equal_frequency_discretization: 'equal_frequency',
                   TIRP.equal_width_discretization: 'equal_width'}

    raw_test_data = data.load(data.datasets_path, raw_test_data_df)

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
                        # inject in each possible way.
                        # labels will be used for creation of expected labels for the RF.
                        test_data = raw_test_data.copy()
                        anomalous_data, labels = inject_to_raw_data(test_data, injection_length, step_over,
                                                                    percentage,
                                                                    epsilon)
                        for method in binning:
                            for number_of_bins in bins:
                                for window_size in window_sizes:
                                    # discover events in separate windows.
                                    test_path_sliding_windows = test_sets_base_folder + '\\KL\\test_events\\{}_{}_{}_{}_{}_{}'.format(
                                        method, number_of_bins, window_size, injection_length, step_over,
                                        percentage)

                                    # make sure dirs exists.
                                    dir_path = test_sets_base_folder + '\\KL\\test_events'
                                    if not os.path.exists(dir_path):
                                        Path(dir_path).mkdir(parents=True, exist_ok=True)

                                    # discover TIRPs.
                                    # pass symbols and entities that were previously found.
                                    suffix = '\\{}_{}_{}'.format(func_2_name[name_2_func[method]],
                                                                 number_of_bins, window_size)

                                    symbols_path = TIRP.input.KL_symbols + suffix
                                    entities_path = TIRP.input.KL_entities + suffix

                                    with open(symbols_path, mode='rb') as syms_path:
                                        ready_symbols = pickle.load(syms_path)
                                    with open(entities_path, mode='rb') as ent_path:
                                        ready_entities = pickle.load(ent_path)

                                    TIRP.make_input(anomalous_data, name_2_func[method], number_of_bins,
                                                    window_size, consider_last=True, regs_to_use=data.most_used,
                                                    test_path=test_path_sliding_windows,
                                                    ready_symbols=ready_symbols, ready_entities=ready_entities)

                                    # create the labels for the detection.
                                    # for each window: [start, end]
                                    test_labels = []
                                    for i in range(len(anomalous_data) - window_size + 1):
                                        # get the labels for the windows' packets.
                                        window_labels = labels[i, i + window_size]

                                        # the label is 0 for a benign packet and 1 for an anomalous packets.
                                        # so a set of packets has an anomaly in it iff the max of its corresponding labels is 1.
                                        window_label = max(window_labels)

                                        # add label.
                                        test_labels.append(window_label)

                                    path = test_sets_base_folder + '\\KL\\test_labels\\{}_{}_{}_{}_{}_{}'.format(
                                        method,
                                        number_of_bins,
                                        window_size,
                                        injection_length,
                                        step_over,
                                        percentage)

                                    dir_path = test_sets_base_folder + '\\KL\\test_labels'
                                    if not os.path.exists(dir_path):
                                        Path(dir_path).mkdir(parents=True, exist_ok=True)

                                    with open(path, mode='wb') as labels_path:
                                        pickle.dump(test_labels, labels_path)


# 1. after running KL on the test_events, filter TIRPs by support.
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
    binning, bins, windows, epsilons, max_gaps, min_horizontal_supps_percentiles = get_KL_params(KL_config_file_path)

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
                            windows_folders_folder = KL_test_sets_base + "\\{}_{}_{}_{}_{}_{}".format(binning, bins,
                                                                                                      window,
                                                                                                      injection_length,
                                                                                                      step_over,
                                                                                                      percentage)

                            max_gaps_times_supports = itertools.product(max_gaps, min_horizontal_supps_percentiles)
                            KL_hyperparams = itertools.product(epsilons, max_gaps_times_supports)

                            # parameters of KL.
                            for eps_gap_supp in KL_hyperparams:
                                epsilon = eps_gap_supp[0]
                                max_gap = eps_gap_supp[1][0]
                                horizontal_supp_percentile = eps_gap_supp[1][1]

                                # we will save the filtered TIRPs here.
                                dest_path_suffix = "\\{}_{}_{}".format(epsilon, horizontal_supp_percentile, max_gap)
                                # path to read the whole set of TIRPs from.
                                src_path_suffix = "\\{}_{}_{}".format(epsilon, default_supp, max_gap)

                                windows_outputs_destination_folder_path = windows_folders_folder + dest_path_suffix
                                windows_outputs_src_folder_path = windows_folders_folder + src_path_suffix

                                if not os.path.exists(windows_outputs_destination_folder_path):
                                    Path(windows_outputs_destination_folder_path).mkdir(parents=True)

                                # call helper function to finish.
                                filter_and_write(windows_outputs_src_folder_path,
                                                 windows_outputs_destination_folder_path, horizontal_supp_percentile)


# 2. after filtering, make dfs for the LSTM.
def create_test_sets_KLSTM(KL_config_path, injections_config_path):
    # 1. go over kl params
    binning_methods, bins, windows, epsilons, max_gaps, min_horizontal_supps_percentiles = get_KL_params(KL_config_path)

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

                            test_windows_folders_folder = KL_test_sets_base + "\\{}_{}_{}_{}_{}_{}".format(binning,
                                                                                                           bins,
                                                                                                           window,
                                                                                                           injection_length,
                                                                                                           step_over,
                                                                                                           percentage)

                            max_gaps_times_supports = itertools.product(max_gaps, min_horizontal_supps_percentiles)
                            KL_hyperparams = itertools.product(epsilons, max_gaps_times_supports)

                            # parameters of KL.
                            for eps_gap_supp in KL_hyperparams:
                                epsilon = eps_gap_supp[0]
                                max_gap = eps_gap_supp[1][0]
                                horizontal_supp_percentile = eps_gap_supp[1][1]

                                # we will save the filtered TIRPs here.
                                test_path_suffix = "\\{}_{}_{}".format(epsilon, horizontal_supp_percentile, max_gap)

                                # folder of tirps text files.
                                test_windows_outputs_folder_path = test_windows_folders_folder + test_path_suffix

                                # need to get the file of the TIRPs in the matching train set. pass it as tirps path.

                                TIRP_path = TIRPs_base + '\\{}_{}_{}_{}_{}_{}'.format(binning, bins, window,
                                                                                      epsilon,
                                                                                      horizontal_supp_percentile,
                                                                                      max_gap)
                                # call parse outout and save.
                                test_df = TIRP.output.parse_output(test_windows_outputs_folder_path,
                                                                   tirps_path=TIRP_path, train=False)

                                # save df.
                                test_df_path_dir = test_sets_base_folder + '\\KL\\KL_LSTM'

                                if not os.path.exists(test_df_path_dir):
                                    Path(test_df_path_dir).mkdir(parents=True, exist_ok=True)

                                test_df_path = test_df_path_dir + '\\{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(binning,
                                                                                                        bins, window,
                                                                                                        horizontal_supp_percentile,
                                                                                                        epsilon,
                                                                                                        max_gap,
                                                                                                        injection_length,
                                                                                                        step_over,
                                                                                                        percentage)

                                with open(test_df_path, mode='wb') as test_f:
                                    pickle.dump(test_df, test_f)


# WAIT WITH THIS. NEED TO DETERMINE THE DETECTION METHOD.
# 3. after having DFs for the LSTMs, make dfs for OCSVMs
def create_test_df_for_KL_based_OCSVM(KL_config_path, injections_config_path):
    # the labels already exist, we saved them when creating test sets for KL.
    # when testing KLSTMs, save the predictions of the lstm as they will be used as the test set for the ocsvm.
    # no need to do anything here. kept as a reminder.
    return None


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
def test_LSTM_based_classifiers(models_train_config, injection_config, tests_config_path, group=''):
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
    with open(models_train_config, mode='r') as train_config:
        models_folders, data_folders, binning_dict, params = get_models_folders_data_folders(train_config)
        for i in range(len(models_folders)):
            models_folders[i] = group + '_' + models_folders[i]
        for i in range(len(data_folders)):
            data_folders[i] = group + '_' + data_folders[i]
    with open(injection_config, mode='r') as injection_config:
        injection_params = yaml.load(injection_config, Loader=yaml.FullLoader)
    with open(tests_config_path, mode='r') as test_config:
        test_params = yaml.load(test_config, Loader=yaml.FullLoader)['processing_config']

    injection_lengths = injection_params['InjectionLength']
    step_overs = injection_params['StepOver']
    percentages = injection_params['Percentage']
    epsilons = injection_params['Epsilon']

    # 2.
    for model_type in ['OCSVM', 'RF']:
        for prefix in ['', 'diff_']:
            results_df = pd.DataFrame(columns=excel_cols)
            for models_folder in models_folders:
                # a.
                data_version = models_folder.split(sep='_', maxsplit=1)[1]
                binning_method = data_version.split(sep='_', maxsplit=1)[0]
                data_version_dict = test_params[data_version]
                if data_version_dict['use']:
                    for model_folder in os.listdir(data.modeles_path + '\\' + models_folder):
                        model_path = data.modeles_path + "\\" + models_folder + "\\" + model_folder
                        with open(model_folder, mode='rb'):
                            LSTM = tensorflow.keras.models.load_model(model_path)
                        model_name = model_folder + '_{}'.format(model_type)
                        classifiers_dir = models.SCADA_base + '\\{}s\\'.format(model_name) + prefix + models_folder
                        for classifier in os.listdir(classifiers_dir):
                            if model_name in classifier:
                                number_of_bins = model_folder.split(sep='_')[-1]
                                injection_sets_folder = models_folder + '_' + number_of_bins
                                desc = data_version_dict['desc']
                                # b.
                                for injection_length in injection_lengths:
                                    for step_over in step_overs:
                                        anomaly_percentage = injection_length / (injection_length + step_over)
                                        if anomaly_percentage > 0.2:
                                            continue
                                        for percentage in percentages:
                                            for epsilon in epsilons:
                                                p_suffix = '_{}_{}_{}_{}_{}'.format(
                                                    injection_sets_folder, desc, number_of_bins, desc, injection_length,
                                                    step_over, percentage, epsilon)
                                                if group != '':
                                                    p_suffix = group + p_suffix

                                                p_x_test = test_sets_base_folder + '\\LSTM_RF_OCSVM\\{}_{}_{}\\X_test_' + p_suffix
                                                p_y_test = test_sets_base_folder + '\\LSTM_RF_OCSVM\\{}_{}_{}\\y_test_' + p_suffix
                                                p_labels = test_sets_base_folder + '\\LSTM_RF_OCSVM\\{}_{}_{}\\labels_' + p_suffix

                                                with open(p_x_test, mode='rb') as X_test_path:
                                                    X_test = pickle.load(X_test_path)
                                                with open(p_y_test, mode='rb') as Y_test_path:
                                                    y_test = pickle.load(Y_test_path)
                                                with open(p_labels, mode='rb') as labels_path:
                                                    # labels are the labels (1/0) for all the packets.
                                                    # however, the LSTM predicts the 21st packet and onwards.
                                                    labels = pickle.load(labels_path)

                                                pred = LSTM.predict(X_test)
                                                test_labels = labels[20:]
                                                if prefix == 'diff_':
                                                    test = np.abs(pred - y_test)
                                                else:
                                                    test = pred

                                                    # now get the exact RF model.
                                                with open(classifiers_dir + '\\' + classifier,
                                                          mode='rb') as classifier_p:
                                                    trained_classifier = pickle.load(classifier_p)

                                                p_dir = LSTM_classifiers_classifications + '\\' + prefix + model_type

                                                # make classifications.
                                                classifications = trained_classifier.predict(test)

                                                # parameters for excel.
                                                data_version_for_excel = data_version_dict['name']
                                                binning_method_for_excel = binning_method
                                                number_of_bins_for_excel = number_of_bins

                                                # calculate metrics.
                                                precision, recalls, thresholds = precision_recall_curve(
                                                    y_true=test_labels,
                                                    probas_pred=classifications)
                                                precision = precision[0]
                                                recall = recalls[0]
                                                auc_score = roc_auc_score(y_true=test_labels, y_score=classifications)
                                                f1 = f1_score(y_true=test_labels, y_pred=classifications)

                                                result = {'data version': data_version_for_excel,
                                                          'binning': binning_method_for_excel,
                                                          '# bins': number_of_bins_for_excel,
                                                          'precision': precision, 'recall': recall, 'auc': auc_score,
                                                          'f1': f1,
                                                          'injection length': injection_length,
                                                          'step over': step_over,
                                                          'percentage': percentage}
                                                # describe: data version, binning, number of bins, RF params
                                                if not os.path.exists(p_dir):
                                                    Path(p_dir).mkdir(parents=True, exist_ok=True)
                                                if prefix == 'diff_':
                                                    split_model = classifier.split(model_name)[1]
                                                else:
                                                    split_model = classifier.split(model_name)[0]
                                                if model_type == 'RF':
                                                    RF_params_for_excel = split_model.split(sep='_')
                                                    estimators = RF_params_for_excel[1]
                                                    criterion = RF_params_for_excel[3]
                                                    max_feature = RF_params_for_excel[5].split(sep='.')[0]
                                                    result['# estimators'] = estimators
                                                    result['criterion'] = criterion
                                                    result['max features'] = max_feature
                                                    for col_name in excel_cols.difference(RF_cols):
                                                        result[col_name] = '-'
                                                    p = p_dir + '\\{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(data_version,
                                                                                                         binning_method,
                                                                                                         number_of_bins,
                                                                                                         estimators,
                                                                                                         criterion,
                                                                                                         max_feature,
                                                                                                         injection_length,
                                                                                                         step_over,
                                                                                                         percentage,
                                                                                                         epsilon)
                                                else:
                                                    OCSVM_params_for_excel = split_model.split(sep='_')
                                                    nu = OCSVM_params_for_excel[1]
                                                    kernel = OCSVM_params_for_excel[3].split(sep='.')[0]
                                                    result['kernel'] = kernel
                                                    result['nu'] = nu
                                                    for col_name in excel_cols.difference(OCSVM_cols):
                                                        result[col_name] = '-'
                                                    p = p_dir + '\\{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(data_version,
                                                                                                      binning_method,
                                                                                                      number_of_bins,
                                                                                                      kernel, nu,
                                                                                                      injection_length,
                                                                                                      step_over,
                                                                                                      percentage,
                                                                                                      epsilon)
                                                with open(p, mode='wb') as classifications_file:
                                                    pickle.dump(classifications, classifications_file)
                                                results_df = pd.concat([results_df,
                                                                        pd.DataFrame.from_dict(data={'0': result},
                                                                                               orient='index',
                                                                                               columns=excel_cols)],
                                                                       axis=0,
                                                                       ignore_index=True)
                                                with open(test_LSTM_RF_OCSVM_log, mode='a') as test_log:
                                                    test_log.write('recorded results of model from type: {}\n'.format(
                                                        prefix + model_type))
                                                    test_log.write(
                                                        'injection parameters are: len: {}, step: {}, %: {}, eps: {}\n'.format(
                                                            injection_length, step_over, percentage, epsilon))
                                                    test_log.write('model parameters:\n')
                                                    test_log.write(
                                                        'data version: {}, # bins: {}, binning method: {}\n'.format(
                                                            data_version_for_excel, number_of_bins_for_excel,
                                                            binning_method_for_excel))
                                                    if model_type == 'RF':
                                                        test_log.write(
                                                            '# estimators: {}, criterion: {}, max features: {}\n'.format(
                                                                result['# estimators'], result['criterion'],
                                                                result['max features']))
                                                    else:
                                                        test_log.write('kernel: {}, nu: {}\n'.format(result['kernel'],
                                                                                                     result['nu']))
                                                    test_log.write(
                                                        'scores: precision: {}, recall: {}, auc: {}, f1: {}\n'.format(
                                                            result['precision'], result['recall'], result['auc scores'],
                                                            result['f1']))
            best_df = make_best(results_df)

            with pd.ExcelWriter(xl_path) as writer:
                best_df['name'] = prefix + model_type
                results_df['name'] = prefix + model_type
                sheet = prefix + model_type + ' performance.'
                if group != '':
                    sheet += 'group: {}'.format(group)
                results_df.to_excel(excel_writer=writer, sheet_name=sheet)
                sheet = prefix + model_type + ' best scores.'
                if group != '':
                    sheet += 'group: {}'.format(group)
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


#################################IRRELEVANT FOR NOW#####################################################

def test_DFA(injection_config, group=''):
    folders = {"k_means": 'KMeans', "equal_frequency": 'EqualFreq',
               "equal_width": 'EqualWidth'}
    n_bins, binners, names = get_DFA_params()
    injection_lengths, step_overs, percentages, epsilons = get_injection_params(injection_config)

    results_df = pd.DataFrame(columns=excel_cols)
    labels_df = pd.DataFrame(columns=['binning', '# bins', 'injection length', 'step over', 'percentage', 'window size',
                                      '# window', 'model label', 'true label'])

    if group is '':
        middle = '\\DFA'
    else:
        middle = '\\DFA_{}'.format(group)
    for injection_length in injection_lengths:
        for step_over in step_overs:
            anomaly_percentage = injection_length / (injection_length + step_over)
            if anomaly_percentage > 0.2:
                continue
            for percentage in percentages:
                for epsilon in epsilons:
                    for binner in binners:
                        for bins in n_bins:
                            p_x_test = test_sets_base_folder + middle + '\\X_test_{}_{}_{}_{}_{}'.format(
                                names[binner],
                                bins,
                                injection_length,
                                step_over,
                                percentage)
                            p_labels = test_sets_base_folder + middle + '\\labels_{}_{}_{}_{}_{}'.format(
                                names[binner],
                                bins,
                                injection_length,
                                step_over,
                                percentage)
                            p_pkt_dict = test_sets_base_folder + middle + '\\labels_{}_{}_{}_{}_{}'.format(
                                names[binner], bins, injection_length,
                                step_over,
                                percentage)

                            with open(p_x_test, mode='rb') as test_data_path:
                                test_df = pickle.load(test_data_path)
                            with open(p_labels, mode='rb') as labels_path:
                                test_labels = pickle.load(labels_path)
                            with open(p_pkt_dict, mode='rb') as pkts_to_stats_p:
                                pkts_to_states_dict = pickle.load(pkts_to_stats_p)

                            with open(data.automaton_path + middle + "\\DFA_{}_{}".format(folders[names[binner]], bins),
                                      mode='rb') as dfa_path:
                                DFA = pickle.load(dfa_path)

                            # the DFA classifies transitions.
                            registers = test_df.columns[2::]
                            start = time.time()
                            decisions = models.automaton.detect(DFA, test_df, registers)
                            end = time.time()

                            elapsed = end - start
                            avg_elapsed = elapsed / len(test_labels)

                            precision, recall, auc_score, f1, prc_auc_score = get_metrics(
                                y_true=test_labels, y_pred=decisions)

                            result = {'binning': names[binner],
                                      '# bins': bins,
                                      'injection length': injection_length,
                                      'step over': step_over,
                                      'percentage': percentage,
                                      'precision': precision,
                                      'recall': recall,
                                      'auc': auc_score,
                                      'f1': f1,
                                      'prc': prc_auc_score}
                            for w in window_sizes:
                                true_windows_labels, dfa_window_labels = DFA_window_labels(decisions, test_labels, w,
                                                                                           pkts_to_states_dict)
                                labels_test_df = pd.DataFrame(columns=labels_df.columns)
                                labels_test_df['# window'] = [i for i in range(len(true_windows_labels))]
                                labels_test_df['window size'] = w
                                labels_test_df['model label'] = dfa_window_labels
                                labels_test_df['true label'] = true_windows_labels
                                labels_test_df['binning'] = folders[names[binner]]
                                labels_test_df['# bins'] = bins
                                labels_test_df['injection length'] = injection_length
                                labels_test_df['step over'] = step_over
                                labels_test_df['percentage'] = percentage
                                labels_df = pd.concat([labels_df, labels_test_df], ignore_index=True)

                            for col_name in excel_cols.difference(DFA_cols):
                                result[col_name] = '-'
                            results_df = pd.concat(
                                [results_df,
                                 pd.DataFrame.from_dict(data={'0': result}, columns=excel_cols, orient='index')],
                                axis=0, ignore_index=True)
                            with open(test_DFA_log, mode='a') as test_log:
                                test_log.write('recorded DFA results for injection with parameters:\n')
                                test_log.write(
                                    'inference: {}, avg inference: {}, binning: {}, # bins: {}'.format(elapsed,
                                                                                                       avg_elapsed,
                                                                                                       names[binner],
                                                                                                       bins))
                                test_log.write('len: {}, step: {}, %: {}\n'.format(injection_length, step_over,
                                                                                   percentage))
                                test_log.write(
                                    'scores: precision: {}, recall: {}, auc: {}, f1: {}, prc:{}, tn: {}, fn : {}, tp: {}, fp: {}\n'.format(
                                        result['precision'],
                                        result['recall'],
                                        result['auc'],
                                        result['f1'], result['prc'], result['tn'], result['fn'], result['tp'],
                                        result['fp']))

    # write to excel.
    best_df = make_best(results_df)
    if group == '':
        algo = 'DFA'
        performance = ''
    else:
        algo = 'DFA_{}'.format(group)
        performance = 'for group {}'.format(group)
    with pd.ExcelWriter(xl_path) as writer:
        results_df['algorithm'] = algo
        results_df.to_excel(excel_writer=writer, sheet_name='DFA ' + performance)
        best_df['algorithm'] = algo
        best_df.to_excel(excel_writer=writer, sheet_name='DFA best ' + performance)
    with pd.ExcelWriter(xl_labels_path) as writer:
        labels_df.to_excel(writer, sheet_name='DFA window labels')


def many_PLC_DFAs(groups_ids, injection_config):
    # Decide on the best parameter configuration. Use a weighted average of the scores.
    best_df = pd.DataFrame(columns=best_cols)
    test_base = test_sets_base_folder
    test_sets_lengths = {}
    weights = {}
    total_samples = 0

    # test all DFAs.
    for group_id in groups_ids:
        test_DFA(injection_config, group_id)

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

    # dfa parameters.
    numbers_of_bins, binners, names = get_DFA_params()

    # injection paramters.
    injection_lengths, step_overs, percentages, epsilons = get_injection_params(injection_config)

    sheets_dfs = dict()

    # get best scores for each group
    for group_id in groups_ids:
        sheet_name = 'DFA best for group ' + group_id
        sheets_dfs[group_id] = pd.read_excel(xl_path, sheet_name)

    for number_of_bins in numbers_of_bins:
        for binner in binners:
            for injection_length in injection_lengths:
                for step_over in step_overs:
                    anomaly_percentage = injection_length / (injection_length + step_over)
                    if anomaly_percentage > 0.2:
                        continue
                    for percentage in percentages:
                        for epsilon in epsilons:
                            # now calculate the scores.
                            method_f1 = 0
                            method_auc = 0
                            method_recall = 0
                            method_precision = 0
                            for group_id in groups_ids:
                                group_best = sheets_dfs[group_id]
                                mask = group_best[
                                           '# bins'] == number_of_bins \
                                       and group_best['binning'] == names[binner] and group_best[
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
                            new_entry = {'data version': '-', 'binning': names[binner],
                                         '# bins': numbers_of_bins,
                                         'precision': method_precision, 'recall': method_recall,
                                         'auc': method_auc, 'f1': method_f1,
                                         'injection length': injection_length,
                                         'step over': step_over, 'percentage': percentage}
                            entry_df = pd.DataFrame.from_dict(columns=best_df.columns,
                                                              data={'0': new_entry}, orient='index')
                            best_df = pd.concat([best_df, entry_df], ignore_index=True)

    # write to xl.
    with pd.ExcelWriter(xl_path) as writer:
        sheet = 'DFA, many PLCs, best scores'
        best_df.to_excel(writer, sheet_name=sheet)


######################IRRELEVANT FOR NOW########################################
def test_KL_based_RF(KL_config_path, injection_config_path):
    results_df = pd.DataFrame(columns=excel_cols)
    with open(KL_config_path, mode='r') as KL_params_path:
        params = yaml.load(KL_params_path, Loader=yaml.FullLoader)['KarmaLegoParams']
        KL_params = params['KarmaLegoParams']
        RF_params = params['RF']

    binning_methods = KL_params['BinningMethods']
    bins = KL_params['Bins']
    windows = KL_params['Windows']
    KL_epsilons = KL_params['Epsilons']
    max_gaps = KL_params['MaxGaps']
    min_ver_sups = KL_params['MinVerSups']
    with open(injection_config_path, mode='r') as anomalies_config:
        injection_params = yaml.load(anomalies_config, Loader=yaml.FullLoader)
    injection_lengths = injection_params['InjectionLength']
    step_overs = injection_params['StepOver']
    percentages = injection_params['Percentage']
    epsilons = injection_params['Epsilon']

    number_of_estimators = RF_params['n_estimators']
    criterions = RF_params['criterion']
    max_features = RF_params['max_features']

    models_base_path = KL_RF_base
    for binning_method in binning_methods:
        for number_of_bins in bins:
            for window in windows:
                TIRP_level = "\\{}_bins_{}_window{}".format(binning_method, number_of_bins, window)
                for KL_epsilon in KL_epsilons:
                    for max_gap in max_gaps:
                        for min_ver_sup in min_ver_sups:
                            path_suffix = "\\eps_{0}_minVS_{1}_maxGap_{2}_HS_true".format(KL_epsilon, min_ver_sup,
                                                                                          max_gap)
                            models_folder = models_base_path + TIRP_level + path_suffix
                            for estimators in number_of_estimators:
                                for criterion in criterions:
                                    for max_feature in max_features:
                                        RF_level = 'estimators_{}_'.format(
                                            estimators) + 'criterion{}_'.format(
                                            criterion) + 'features_{}.sav'.format(max_feature)
                                        RF_classifier_path = models_folder + '\\' + RF_level
                                        with open(RF_classifier_path, mode='rb') as path:
                                            RF_classifier = pickle.load(path)
                                        input_base = data.datasets_path + '\\KL\\'
                                        for injection_length in injection_lengths:
                                            for step_over in step_overs:
                                                anomaly_percentage = injection_length / (injection_length + step_over)
                                                if anomaly_percentage > 0.2:
                                                    continue
                                                for percentage in percentages:
                                                    for injection_epsilon in epsilons:
                                                        desc = "\\{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(binning_method,
                                                                                                      number_of_bins,
                                                                                                      window,
                                                                                                      injection_length,
                                                                                                      step_over,
                                                                                                      percentage,
                                                                                                      injection_epsilon)
                                                        outDir = KL_output_base + desc
                                                        # the path to the dir containing the output of the specific KL for the specific injection.
                                                        # the inputs were the events of the sliding windows in the injected data.
                                                        # call parse output and replace anomaly = 0 with labels.
                                                        windows_outputs_dir = outDir + "\\{0}_{1}_{2}_true".format(
                                                            KL_epsilon,
                                                            min_ver_sup,
                                                            max_gap)
                                                        # call parse_output, get the tirps in the train set for the indexing.
                                                        TIRP_path = TIRPs_base + '\\{}_{}_{}_{}_{}_{}'.format(
                                                            binning_method,
                                                            bins,
                                                            window,
                                                            injection_epsilon,
                                                            min_ver_sup,
                                                            max_gap)
                                                        test_df = models.TIRP.output.parse_output(windows_outputs_dir,
                                                                                                  TIRP_path,
                                                                                                  train=False)

                                                        path_to_labels = test_sets_base_folder + '\\KL\\RF\\test_labels\\{}_{}_{}_{}_{}_{}_{}'.format(
                                                            binning_method,
                                                            number_of_bins,
                                                            window,
                                                            injection_length,
                                                            step_over,
                                                            percentage,
                                                            injection_epsilon)
                                                        with open(path_to_labels, mode='rb') as labels_path:
                                                            test_labels = pickle.load(labels_path)

                                                        classifications = RF_classifier.predict(test_df)

                                                        # calculate metrics.
                                                        precision, recalls, thresholds = precision_recall_curve(
                                                            y_true=test_labels,
                                                            probas_pred=classifications)
                                                        precision = precision[0]
                                                        recall = recalls[0]
                                                        auc_score = roc_auc_score(y_true=test_labels,
                                                                                  y_score=classifications)

                                                        f1 = f1_score(y_true=test_labels, y_pred=classifications)
                                                        result = {'binning': binning_method,
                                                                  '# bins': number_of_bins,
                                                                  'window size': window,
                                                                  'KL epsilon': KL_epsilon,
                                                                  'minimal VS': min_ver_sup,
                                                                  'max gap': max_gap,
                                                                  'injection length': injection_length,
                                                                  'step over': step_over,
                                                                  'percentage': percentage,
                                                                  'precision': precision,
                                                                  'recall': recall,
                                                                  'auc': auc_score,
                                                                  'f1': f1}
                                                        for col_name in excel_cols.difference(KL_based_RF_cols):
                                                            result[col_name] = '-'
                                                        temp_df = pd.DataFrame.from_dict(columns=excel_cols,
                                                                                         data={'0': result},
                                                                                         orient='index')
                                                        results_df = pd.concat([results_df, temp_df], axis=0,
                                                                               ignore_index=True)
                                                        with open(test_KL_RF_log, mode='a') as test_log:
                                                            test_log.write(
                                                                'recorded results for KL-RF for injection with parameters:\n')
                                                            test_log.write('len: {}, step: {}, %: {}, eps: {}\n'.format(
                                                                injection_length, step_over, percentage,
                                                                injection_epsilon))
                                                            test_log.write('model parameters:\n')
                                                            test_log.write(
                                                                'binning: {}, # bins: {}, window size: {}, KL epsilon: {}, minimal VS: {}, max gap: {}\n'.format(
                                                                    binning_method, number_of_bins, window,
                                                                    KL_epsilon, min_ver_sup, max_gap))
                                                            test_log.write(
                                                                'scores: precision: {}, recall: {}, auc: {}, f1: {}\n'.format(
                                                                    result['precision'], result['recall'],
                                                                    result['auc scores'], result['f1']))

    best_df = make_best(results_df)

    with pd.ExcelWriter(xl_path) as writer:
        best_df['name'] = 'KL-RF'
        results_df['name'] = 'KL-RF'
        best_df.to_excel(excel_writer=writer, sheet_name='KL based RF best scores')
        results_df.to_excel(excel_writer=writer, sheet_name='KL based RF performance')


######################IRRELEVANT FOR NOW########################################


# functions for training LSTMs.
def train_LSTM(train_config):
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
    raw_df = data.load(data.datasets_path, 'TRAIN')

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
                suffix = '//{}_{}'.format(method_folder, folder_name) + '//{}'.format(model_name)
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

                dump_model = data.modeles_path + '\\{}_{}'.format(method_folder, folder_name)
                dump_df = data.datasets_path + '\\{}_{}'.format(method_folder, folder_name)
                models.models.simple_LSTM(lstm_input, 20, 42, model_name, train=1.0, models_path=dump_model,
                                          data_path=dump_df)


# for LSTM classifiers.
def create_test_sets_LSTMs(train_config, injection_config):
    folders = {"k_means": 'KMeans', "equal_frequency": 'EqualFreq',
               "equal_width": 'EqualWidth'}

    # injection params.
    injection_lengths, step_overs, percentages, epsilons = get_injection_params(injection_config)

    # first ,inject anomalies. and create the test set for: LSTM , RF(?) and OCSVM(?).
    for injection_length in injection_lengths:
        for step_over in step_overs:
            anomaly_percentage = injection_length / (injection_length + step_over)
            if anomaly_percentage > lim:
                pass
            else:
                for percentage in percentages:
                    for epsilon in epsilons:
                        df_path = test_sets_base_folder + '//raw//data_{}_{}_{}'.format(injection_length, step_over,
                                                                                        percentage)
                        labels_path = test_sets_base_folder + '//raw//labels_{}_{}_{}'.format(injection_length,
                                                                                              step_over,
                                                                                              percentage)
                        with open(df_path, mode='rb') as p:
                            anomalous_data = pickle.load(p)

                        with open(labels_path, mode='rb') as p:
                            labels = pickle.load(p)

                        #  process the data using the different versions, get params.
                        binnings, numbers_of_bins, data_versions = get_LSTM_params(train_config)

                        # folder_name: name of binning method in the folders (KMeans), method_name: name of binning method in files (kmeans)
                        for data_version in data_versions:
                            name = data_version['name']
                            desc = data_version['desc']
                            processed = None
                            if not data_version['reprocess']:
                                processed = data.process(anomalous_data, name, None, None, False)

                            for method_name in binnings:
                                folder_name = folders[method_name]
                                for number_of_bins in numbers_of_bins:
                                    if data_version['reprocess']:
                                        lstm_input = data.process(anomalous_data, name, number_of_bins,
                                                                  method_name,
                                                                  True)
                                    else:
                                        lstm_input = processed.copy()
                                        cols_not_to_bin = data_version['no_bin']
                                        method_folder = folders[method_name]
                                        file_name = data_version['desc']
                                        folder_name = data_version['name']

                                        model_name = '{}_{}_{}'.format(file_name, method_name, number_of_bins)
                                        suffix = '//{}_{}'.format(method_folder, folder_name) + '//{}'.format(
                                            model_name)

                                        # if file_name == 'v1_single_plc_make_entry_v1_20_packets':
                                        # suffix = '//{}_{}'.format(method_folder,
                                        # folder_name) + '//{}_{}_{}'.format(file_name,
                                        # file_name,
                                        # number_of_bins)

                                        # scale everything, bin by config file.
                                        for col_name in lstm_input.columns:
                                            if 'time' not in col_name and 'state' not in col_name and col_name not in cols_not_to_bin:
                                                with open(binners_base + suffix + '_{}'.format(col_name),
                                                          mode='rb') as binner_p:
                                                    binner = pickle.load(binner_p)
                                                lstm_input[col_name] = binner.transform(
                                                    lstm_input[col_name].to_numpy().reshape(-1, 1))
                                            lstm_input[col_name] = data.scale_col(lstm_input, col_name, None)

                                    # now create test data set for LSTM. Only need X_test and y_test.
                                    X_test, y_test = models.custom_train_test_split(
                                        lstm_input,
                                        20, 42, train=1.0)

                                    # now save, X_test, y_test and the labels which will be used to obtain the y_test of the classifier.
                                    p_suffix = '_{}_{}_{}_{}_{}'.format(
                                        folder_name, name, number_of_bins, desc, injection_length,
                                        step_over, percentage, epsilon)
                                    # if group_id != '':
                                    # p_suffix = group_id + p_suffix

                                    # make sure dirs exist and dump.
                                    dir_path = test_sets_base_folder + '\\LSTM\\{}_{}_{}'.format(
                                        folder_name, name, number_of_bins)

                                    bin_part = folders[method_name]

                                    suffix = '{}_{}_{}_{}_{}_{}'.format(folder_name,
                                                                        bin_part,
                                                                        number_of_bins,
                                                                        injection_length,
                                                                        step_over,
                                                                        percentage)

                                    p_x_test = test_sets_base_folder + '//LSTM//X_' + suffix
                                    p_y_test = test_sets_base_folder + '//LSTM//y_' + suffix
                                    p_labels = test_sets_base_folder + '//LSTM//labels_' + suffix

                                    if not os.path.exists(dir_path):
                                        Path(dir_path).mkdir(parents=True, exist_ok=True)

                                    with open(p_x_test, mode='wb') as data_path:
                                        pickle.dump(X_test, data_path)

                                    with open(p_y_test, mode='wb') as data_path:
                                        pickle.dump(y_test, data_path)

                                    with open(p_labels, mode='wb') as data_path:
                                        pickle.dump(labels, data_path)


def test_LSTM(train_config, raw_test_data):
    folders = {"k_means": 'KMeans', "equal_frequency": 'EqualFreq',
               "equal_width": 'EqualWidth'}

    test_df = data.load(test_sets_base_folder, raw_test_data)

    results_df = pd.DataFrame(columns=['data_version', 'binning', '# bins', 'mse', 'r2'])

    # go over all combinations, process raw test set, test and save metric scores.
    binning_methods, numbers_of_bins, data_versions = get_LSTM_params(train_config)

    for data_version in data_versions:
        test_lstm = None
        file_name = data_version['desc']
        folder_name = data_version['name']

        if not data_version['reprocess']:
            test_lstm = data.process(test_df, data_version['name'], None, None, False)
        for binning_method in binning_methods:
            method_folder = folders[binning_method]
            for number_of_bins in numbers_of_bins:
                if data_version['reprocess']:
                    lstm_in = data.process(test_df, data_version['name'], number_of_bins, binning_method, True)
                else:
                    lstm_in = test_lstm.copy()
                    cols_not_to_bin = data_version['no_bin']

                    model_name = '{}_{}_{}'.format(file_name, binning_method, number_of_bins)
                    suffix = '//{}_{}'.format(method_folder, folder_name) + '//{}'.format(model_name)

                    """if file_name == 'v1_single_plc_make_entry_v1_20_packets':
                        suffix = '//{}_{}'.format(method_folder, folder_name) + '//{}_{}_{}'.format(file_name,
                                                                                                    file_name,
                                                                                                    number_of_bins)"""

                    # scale everything, bin by config file.
                    for col_name in lstm_in.columns:
                        if 'time' not in col_name and 'state' not in col_name and col_name not in cols_not_to_bin:
                            with open(binners_base + suffix + '_{}'.format(col_name), mode='rb') as binner_p:
                                binner = pickle.load(binner_p)
                            lstm_in[col_name] = binner.transform(lstm_in[col_name].to_numpy().reshape(-1, 1))
                        lstm_in[col_name] = data.scale_col(lstm_in, col_name, None)

                bin_part = folders[binning_method]
                version_part = data_version['name']
                model_name = '{}_{}_{}'.format(data_version['desc'], binning_method, number_of_bins)
                dump_model = data.modeles_path + '\\{}_{}'.format(bin_part, version_part)

                with open(dump_model + '\\' + model_name, mode='rb') as model_path:
                    LSTM = keras.models.load_model(model_path)

                X_test, y_test = models.models.custom_train_test_split(lstm_in, 20, 42, train=1.0)

                dir_p = test_sets_base_folder + '//{}_{}'.format(bin_part, version_part)
                if not os.path.exists(dir_p):
                    Path(dir_p).mkdir(parents=True, exist_ok=True)

                with open(dir_p + '//X_test_' + model_name) as x_test_p:
                    pickle.dump(X_test, x_test_p)

                with open(dir_p + '//y_test_' + model_name) as y_test_p:
                    pickle.dump(y_test, y_test_p)

                y_pred = LSTM.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                result = {'data version': data_version['name'],
                          'binning': folders[binning_method],
                          '# bins': number_of_bins,
                          'mse': mse,
                          'r2': r2}

                res_df = pd.DataFrame.from_dict(columns=results_df.columns, data={'0': result}, orient='index')
                results_df = pd.concat([results_df, res_df], ignore_index=True)
                with open(lstm_test_log, mode='a') as log:
                    log.write('mse:{}, r2:{}, version:{}, binning:{}, bins:{}'.format(mse, r2, data_version['name'],
                                                                                      folders[binning_method],
                                                                                      number_of_bins))

    with pd.ExcelWriter(xl_path) as writer:
        results_df.to_excel(writer, sheet_name='LSTM scores')


def detect_LSTM(lstm_config, injection_config, d='L2', t='# std'):
    injection_lengths, step_overs, percentages, injection_epsilons = get_injection_params(injection_config)

    # grid over the lstm data sets : get labels and data.
    # for each data set: predict and get metrics.

    folders = {"k_means": 'KMeans', "equal_frequency": 'EqualFreq',
               "equal_width": 'EqualWidth'}

    results_df = pd.DataFrame(columns=excel_cols)
    labels_df = pd.DataFrame(
        columns=['data version', 'binning', '# bins', '#std', 'p_value', 'distance metric', 'injection length', 'percentage',
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
                    deviations = calc_MD(pred, val_y, cov_val_inv)  # Mahalanobis distance of prediction from ground truth.
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
                                            pred_labels = [1 if calc_p_value(mean, std, dist) < limit else 0 for dist in test_deviations]

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


def create_val_for_LSTM(lstm_config):
    folders = {"k_means": 'KMeans', "equal_frequency": 'EqualFreq',
               "equal_width": 'EqualWidth'}

    val_df = data.load(data.datasets_path, 'MB_TCP_VAL')

    # go over all combinations, process raw test set, test and save metric scores.
    binning_methods, numbers_of_bins, data_versions = get_LSTM_params(lstm_config)

    for data_version in data_versions:
        val_lstm = None
        folder_name = data_version['name']
        file_name = data_version['desc']
        if not data_version['reprocess']:
            val_lstm = data.process_data_v3(val_df, None, None, False)
        for binning_method in binning_methods:
            method_folder = folders[binning_method]
            for number_of_bins in numbers_of_bins:
                if data_version['reprocess']:
                    lstm_in = data.process(val_df, number_of_bins, binning_method, True)
                else:
                    lstm_in = val_lstm.copy()
                    cols_not_to_bin = data_version['no_bin']
                    model_name = '{}_{}_{}'.format(file_name, binning_method, number_of_bins)
                    suffix = '//{}_{}'.format(method_folder, folder_name) + '//{}'.format(model_name)

                    # scale everything, bin by config file.
                    for col_name in lstm_in.columns:
                        if 'time' not in col_name and 'state' not in col_name and col_name not in cols_not_to_bin:
                            with open(binners_base + suffix + '_{}'.format(col_name), mode='rb') as binner_p:
                                binner = pickle.load(binner_p)
                            lstm_in[col_name] = binner.transform(lstm_in[col_name].to_numpy().reshape(-1, 1))
                        lstm_in[col_name] = data.scale_col(lstm_in, col_name, None)

                X_val, y_val = models.custom_train_test_split(lstm_in, series_len=20, np_seed=42, train=1.0)

                validation_path_X = LSTM_validation + '{}_{}\\X_{}_{}_{}'.format(folder_name, folders[binning_method],
                                                                                 file_name, binning_method,
                                                                                 number_of_bins)
                validation_path_Y = LSTM_validation + '{}_{}\\X_{}_{}_{}'.format(folder_name, folders[binning_method],
                                                                                 file_name, binning_method,
                                                                                 number_of_bins)
                with open(validation_path_X, mode='wb') as x_val_f:
                    pickle.dump(X_val, x_val_f)
                with open(validation_path_Y, mode='wb') as y_val_f:
                    pickle.dump(y_val, y_val_f)


def create_raw_test_sets(injections_config):
    # get params for injections.
    injection_lengths, step_overs, percentages, epsilons = get_injection_params(injections_config)

    with open(test_sets_base_folder + '//MB_TCP_TEST', mode='rb') as test_path:
        test_data = pickle.load(test_path)

    for injection_length in injection_lengths:
        for step_over in step_overs:
            anomaly_percentage = injection_length / (injection_length + step_over)
            if anomaly_percentage > lim:
                pass
            else:
                for percentage in percentages:
                    for epsilon in epsilons:
                        anomalous_data, labels = inject_to_raw_data(test_data, injection_length, step_over,
                                                                    percentage,
                                                                    epsilon)
                        df_path = test_sets_base_folder + '//raw//data_{}_{}_{}'.format(injection_length, step_over,
                                                                                        percentage)
                        labels_path = test_sets_base_folder + '//raw//labels_{}_{}_{}'.format(injection_length,
                                                                                              step_over,
                                                                                              percentage)
                        with open(df_path, mode='wb') as df_p:
                            pickle.dump(anomalous_data, df_p)

                        with open(labels_path, mode='wb') as labels_p:
                            pickle.dump(labels, labels_p)


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


def DFA_window_labels(decisions, true_labels, window_size, pkts_to_states_dict):
    dfa_windows_labels = []
    true_windows_labels = []
    length = len(true_labels)  # true labels is labels of PACKETS (injected / benign).
    for i in range(length - window_size + 1):
        true_window_label = max(true_labels[i: i + window_size])
        window_start_state = pkts_to_states_dict[i]
        window_end_state = pkts_to_states_dict[i + window_size - 1]
        dfa_window_label = max(decisions[window_start_state: window_end_state + 1])
        true_windows_labels.append(true_window_label)
        dfa_windows_labels.append(dfa_window_label)

    return true_windows_labels, dfa_windows_labels


# WHEN COMPARING METHODS TO KLSTM, NEED TO TAKE THE LABELS FROM THE 21ST WINDOW FOR THE OTHER METHOD.
def KLSTM_window_labels(model_windows_labels, true_labels, window_size):
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
        KL_params = yaml.load(KL_params_path, Loader=yaml.FullLoader)['KarmaLegoParams']

    binning_methods = KL_params['BinningMethods']
    bins = KL_params['Bins']
    windows = KL_params['Windows']
    epsilons = KL_params['Epsilons']
    max_gaps = KL_params['MaxGaps']
    min_h_percentile_sups = KL_params['MinHorizontalSups']
    return binning_methods, bins, windows, epsilons, max_gaps, min_h_percentile_sups


def get_DFA_params():
    bins = [5, 6, 7, 8, 9, 10]
    binning_methods = {'k_means': data.k_means_binning, 'equal_frequency': data.equal_frequency_discretization,
                       'equal_width': data.equal_width_discretization}
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

    return precision, recall, auc_score, f1, prc_auc_score


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

