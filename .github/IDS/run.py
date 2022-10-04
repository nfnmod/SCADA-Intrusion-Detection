import csv
import itertools
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow
import yaml

import data
import models
import models.TIRP as TIRP

from data.injections import inject_to_raw_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score

KL_base = data.datasets_path + "\\KL\\"
KL_RF_base = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\KL RF'
HTM_base = "C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\HTM\\"
logs = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\log files\\'
test_sets_base_folder = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\test sets'
KL_based_RF_log = logs + 'KarmaLego based RF.txt'
DFA_log = logs + 'DFA.txt'
KL_output_base = "C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\test sets\\KL\\KL out"

DFA_regs = ['30', '120', '15']

excel_cols = {'data version', 'binning', '# bins', 'nu', 'kernel', '# estimators', 'criterion', 'max features',
              'ON bits', 'SDR size', 'numOfActiveColumnsPerInhArea', 'potential Pct', 'synPermConnected',
              'synPermActiveInc', 'synPermInactiveDec', 'boostStrength', 'cellsPerColumn', 'newSynapseCount',
              'initialPerm', 'permanenceInc', 'permanenceDec', 'maxSynapsesPerSegment', 'maxSegmentsPerCell',
              'minThreshold', 'activationThreshold', 'window size', 'KL epsilon', 'minimal VS', 'max gap',
              'injection length', 'step over', 'injection epsilon', 'percentage', 'precision', 'recall', 'auc', 'f1'}
RF_cols = {'data version', 'binning', '# bins', '# estimators', 'criterion', 'max features',
           'injection length', 'step over', 'injection epsilon', 'percentage', 'precision', 'recall', 'auc', 'f1'}
OCSVM_cols = {'data version', 'binning', '# bins', 'nu', 'kernel',
              'injection length', 'step over', 'injection epsilon', 'percentage', 'precision', 'recall', 'auc', 'f1'}

DFA_cols = {'injection length', 'step over', 'injection epsilon', 'percentage', 'precision', 'recall', 'auc', 'f1'}

KL_based_RF_cols = {'binning', '# bins', 'window size', 'KL epsilon', 'minimal VS', 'max gap',
                    'injection length', 'step over', 'injection epsilon', 'percentage', 'precision', 'recall', 'auc',
                    'f1'}

xl_path = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\excel\\classifiers comprison.xlsx'

test_LSTM_RF_OCSVM_log = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\log files\\test LSTM-RF-OCSVM.txt'
test_DFA_log = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\log files\\test DFA.txt'
test_KL_RF_log = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\log files\\test KL-RF.txt'


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


def train_RF(RF_train_config_file_path):
    with open(RF_train_config_file_path, mode='r') as train_config:
        models_folders, data_folders, binning_dict, params = get_models_folders_data_folders(train_config)
        zipped = zip(models_folders, data_folders)
        for folder_pair in zipped:
            models_folder = folder_pair[0]
            data_folder = folder_pair[1]
            binning_version = (data_folder.split(sep='_'))[0]
            models.models.make_classifier(models_folder=models_folder, data_folder=data_folder,
                                          binning=binning_dict[binning_version], params=params, RF_only=True)


def train_OCSVM(OCSVM_train_config_file_path):
    with open(OCSVM_train_config_file_path, mode='r') as train_config:
        models_folders, data_folders, binning_dict, params = get_models_folders_data_folders(train_config)
        zipped = zip(models_folders, data_folders)
        for folder_pair in zipped:
            models_folder = folder_pair[0]
            data_folder = folder_pair[1]
            binning_version = (data_folder.split(sep='_'))[0]
            models.models.make_classifier(models_folder=models_folder, data_folder=data_folder,
                                          binning=binning_dict[binning_version], params=params, OCSVM_only=True)


def train_automaton():
    pkts = data.load(data.datasets_path, "modbus")
    with open(DFA_log, mode='a') as log:
        log.write('Creating DFA')
        start = time.time()
        DFA = models.automaton.make_automaton(data.to_bin, pkts)
        end = time.time()
        log.write('Done, time elapsed:{}'.format(end - start))
        data.dump(data.automaton_path, "DFA", DFA)


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
        windows = discovery_params['windows_sizes']
        bins_window_options = itertools.product(number_of_bins, windows)
        options = itertools.product(binning, bins_window_options)
        for option in options:
            b = binning_methods[option[0]]
            k = option[1][0]
            w = option[1][1]
            TIRP.discover(plc_df, b, k, w, consider_last=True, stats_dict=stats_dict)
            TIRP.make_input(plc_df, b, k, w, stats_dict, consider_last=True)


def train_RF_from_KL(KL_config_file_path):
    """
     go over all pairs of all_TIRPs_path, window_TIRPs_folder:
        for each binning, bins, window:
             folder = KL_base + "whole_out\\all_{binning}_{bins}_{window} " (WHOLE TIRPS FOLDER)
             folder2 = KL_base + "{binning}_{bins}_{window}_out" (WINDOWS FOLDERS)
             for epsilon, max gap, vertical support:
                path = folder + \\ epsilon_max_gap_support_hs.txt (output for the TIRPs discovery phase by KL with the parameters)
                path2 = folder2 + "epsilon_max_gap_support_hs" (outputs of KL on sliding windows
                                                                                        with the parameters)
                run parse_output(path, path2)
                train RF classifier with the df returned
                save rf
    """
    with open(KL_config_file_path, mode='r') as train_config:
        params = yaml.load(train_config, Loader=yaml.FullLoader)
        KL_params = params['KarmaLegoParams']
        binning = KL_params['BinningMethods']
        bins = KL_params['Bins']
        windows = KL_params['Windows']
        epsilons = KL_params['Epsilons']
        max_gaps = KL_params['MaxGaps']
        min_ver_supps = KL_params['MinVerSups']
        binning_times_bins = itertools.product(binning, bins)
        parent_folders = itertools.product(windows, binning_times_bins)
        for window_binning_bins in parent_folders:
            window = window_binning_bins[0]
            binning = window_binning_bins[1][0]
            bins = window_binning_bins[1][1]
            whole_TIRPs_folder = KL_base + "whole_out\\all_{}_bins_{}_window_{}".format(binning, bins, window)
            windows_folders_folder = KL_base + "{}_bins_{}_window_{}_out".format(binning, bins, window)
            max_gaps_times_supports = itertools.product(max_gaps, min_ver_supps)
            KL_hyperparams = itertools.product(epsilons, max_gaps_times_supports)
            for eps_gap_supp in KL_hyperparams:
                epsilon = eps_gap_supp[0]
                max_gap = eps_gap_supp[1][0]
                ver_supp = eps_gap_supp[1][1]
                path_suffix = "\\eps_{0}_minVS_{1}_maxGap_{2}_HS_true".format(epsilon, ver_supp, max_gap)
                whole_TIRPS_output_path = whole_TIRPs_folder + path_suffix + ".txt"
                windows_outputs_folder_path = windows_folders_folder + path_suffix
                TIRP_df = TIRP.output.parse_output(whole_TIRPS_output_path, windows_outputs_folder_path)
                windows_features = TIRP_df[:, :-1]
                windows_labels = TIRP_df[:, -1]
                X_train, X_test, y_train, y_test = train_test_split(windows_features, windows_labels, test_size=0.2, random_state=42)
                RF_params = params['RF']
                parameters_combinations = itertools.product(RF_params['criterion'], RF_params['max_features'])
                parameters_combinations = itertools.product(RF_params['n_estimators'], parameters_combinations)
                for combination in parameters_combinations:
                    estimators = combination[0]
                    criterion = combination[1][0]
                    max_features = combination[1][1]
                    model = RandomForestClassifier(n_estimators=estimators, criterion=criterion,
                                                   max_features=max_features)
                    with open(KL_based_RF_log, mode='a') as log:
                        log.write('Training KL based RF with:\n')
                        log.write('window size: {} binning: {} bins: {}\n'.format(window, binning, bins))
                        log.write('eps: {}, max_gap: {}, min_ver_supp: {}\n'.format(epsilon, max_gap, ver_supp))
                        log.write('estimators: {}, criterion: {} ,max_features: {}\n'.format(estimators, criterion,
                                                                                             max_features))
                        start = time.time()
                        model.fit(X_train, y_train)
                        end = time.time()
                        models_base_path = KL_RF_base
                        TIRP_level = "\\{}_bins_{}_window{}".format(binning, bins, window)
                        KL_level = path_suffix
                        models_folder = models_base_path + TIRP_level + KL_level
                        tensorflow.keras.models.save_model(model,
                                                           models_folder + '\\' + 'estimators_{}_'.format(
                                                               estimators) + 'criterion{}_'.format(
                                                               criterion) + 'features_{}.sav'.format(max_features))
                        log.write('Done, time elapsed:{}\n'.format(end - start))


# process the raw data using some method without binning but with scaling. Then split the data, convert to csv and save.
# The csv file is formatted by the requirements of HTM.
def create_data_for_HTM(HTM_input_creation_config):
    """
    open file, for each data version:
                    go over all bins, methods combinations and apply data processing.
                    save the train and test sets.
    """

    pkt_df = data.load(data.datasets_path, "modbus")
    with open(HTM_input_creation_config, mode='r') as input_config:
        params = yaml.load(input_config, Loader=yaml.FullLoader)
        versions_dicts = params['processing_config']
        for version_dict in versions_dicts:
            data_version = version_dict['name']
            use = version_dict['use']
            if not use:
                pass
            else:
                processed_df = data.process(pkt_df, data_version, None, None)
                # X_train will be used to train the HTM network. X_test and sets created by injecting anomalies into X_test will be used
                # for testing the HTM network.
                X_train, X_test = train_test_split(processed_df, test_size=0.2, random_state=42)
                # 1. write column names.
                # 2. write columns data types.
                # 3. write df to csv without the columns names.
                folder = HTM_base + '\\datasets\\' + '{}'.format(data_version)

                train_path_str = folder + '\\' + "X_train_" + data_version + ".csv"
                test_path_str = folder + '\\' + "X_test_" + data_version + ".csv"
                train_path = Path(train_path_str)
                test_path = Path(test_path_str)

                train_path.parent.mkdir(parents=True, exist_ok=True)
                test_path.parent.mkdir(parents=True, exist_ok=True)

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

                with open(test_path_str, 'w', newline='') as test_file:
                    test_writer = csv.writer(test_file)
                    # write the field names.
                    test_cols = list(X_test.columns)
                    test_writer.writerow(test_cols)
                    # write the field types.
                    test_cols_types = ['float'] * len(test_cols)
                    test_writer.writerow(test_cols_types)
                    # use no flags.
                    test_writer.writerow([])
                X_test.to_csv(path_or_buf=test_path, index=False, header=False, mode='a')


"""
The following function create the test files for the various classifiers.
Use configuration file to create them.
"""


def create_test_files_LSTM_RF_and_OCSVM_and_HTM(raw_test_data_df, data_versions_config, injection_config):
    """
    grid over injection params, for each combination : inject anomalies and then process the dataset using all methods.
    """
    test_data = data.load(test_sets_base_folder, raw_test_data_df)
    with open(injection_config, mode='r') as anomalies_config:
        injection_params = yaml.load(anomalies_config, Loader=yaml.FullLoader)
        injection_lengths = injection_params['InjectionLength']
        step_overs = injection_params['StepOver']
        percentages = injection_params['percentage']
        epsilons = injection_params['Epsilon']
        # first ,inject anomalies. and create the test set for: LSTM , RF and OCSVM.
        for injection_length in injection_lengths:
            for step_over in step_overs:
                for percentage in percentages:
                    for epsilon in epsilons:
                        anomalous_data, labels = inject_to_raw_data(test_data, injection_length, step_over, percentage,
                                                                    epsilon)
                        #  process the data using the different versions
                        with open(data_versions_config, mode='r') as processing_config:
                            config = yaml.load(processing_config, Loader=yaml.FullLoader)
                            binnings = config['binning']
                            data_versions = config['processing_config']
                            for folder_name, method_name in binnings:
                                # folder_name: name of binning method in the folders (KMeans), method_name: name of binning method in files (kmeans)
                                for data_version in data_versions:
                                    to_use = data_version['use']
                                    if not to_use:
                                        pass
                                    else:
                                        name = data_version['name']
                                        bins = data_version['bins']
                                        desc = data_version['desc']

                                        # same thing but for HTM. no need to save y_test because HTM predicts anomaly scores to be used by
                                        # the classifiers based on the HTM network.
                                        # no binning for HTM, only scaling.
                                        test_df = data.process(anomalous_data, name, None, None)
                                        p_x_test_HTM = test_sets_base_folder + '\\HTM\\{}\\X_test_{}_{}_{}_{}.csv'.format(
                                            name, injection_length,
                                            step_over, percentage, epsilon)
                                        p_labels_HTM = test_sets_base_folder + '\\HTM\\{}\\labels_{}_{}_{}_{}'.format(
                                            name, injection_length,
                                            step_over, percentage, epsilon)

                                        Path(test_sets_base_folder + '\\HTM\\{}'.format(name)).mkdir(parents=True, exist_ok=True)

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

                                        with open(p_labels_HTM, mode='w') as labels_path:
                                            pickle.dump(labels, labels_path)

                                        # now same thing for LSTM, OCSVM.
                                        for number_of_bins in bins:
                                            test_df = data.process(anomalous_data, name, number_of_bins, method_name)
                                            # now create test data set for LSTM. Only need X_test and y_test.
                                            X_train, X_test, y_train, y_test = models.custom_train_test_split(test_df,
                                                                                                              20, 42, train=0.0)
                                            # now save, X_test, y_test and the labels which will be used to obtain the y_test of the classifier.
                                            p_x_test = test_sets_base_folder + '\\LSTM_RF_OCSVM\\{}_{}_{}\\X_test_{}_{}_{}_{}_{}'.format(
                                                folder_name, name, number_of_bins, desc, injection_length,
                                                step_over, percentage, epsilon)
                                            p_y_test = test_sets_base_folder + '\\LSTM_RF_OCSVM\\{}_{}_{}\\y_test_{}_{}_{}_{}_{}'.format(
                                                folder_name, name, number_of_bins, desc, injection_length,
                                                step_over, percentage, epsilon)
                                            p_labels = test_sets_base_folder + '\\LSTM_RF_OCSVM\\{}_{}_{}\\labels_{}_{}_{}_{}_{}'.format(
                                                folder_name, name, number_of_bins, desc, injection_length,
                                                step_over, percentage, epsilon)

                                            # make sure dirs exist and dump.
                                            Path(test_sets_base_folder + '\\LSTM_RF_OCSVM\\{}_{}_{}'.format(folder_name, name, number_of_bins)).mkdir(parents=True, exist_ok=True)

                                            with open(p_x_test, mode='wb') as data_path:
                                                pickle.dump(X_test, data_path)
                                            with open(p_y_test, mode='wb') as data_path:
                                                pickle.dump(y_test, data_path)
                                            with open(p_labels, mode='wb') as data_path:
                                                pickle.dump(labels, data_path)


def create_test_files_DFA(raw_test_data_df, injection_config):
    """
    only one data version so just grid over all the injection parameters.
    """
    test_data = data.load(data.datasets_path, raw_test_data_df)
    with open(injection_config, mode='r') as anomalies_config:
        injection_params = yaml.load(anomalies_config, Loader=yaml.FullLoader)
        injection_lengths = injection_params['InjectionLength']
        step_overs = injection_params['StepOver']
        percentages = injection_params['Percentage']
        epsilons = injection_params['Epsilon']
        # first ,inject anomalies. and create the test set for: LSTM , RF and OCSVM.
        for injection_length in injection_lengths:
            for step_over in step_overs:
                for percentage in percentages:
                    for epsilon in epsilons:
                        anomalous_data, labels = inject_to_raw_data(test_data, injection_length, step_over, percentage,
                                                                    epsilon)
                        test_df = data.process(anomalous_data, 'v3', None, None)

                        # we need to know which transitions include anomalies to create the test labels.
                        # iterate over anomalous_data, if a packet is anomalous, mark the transition from its' corresponding
                        # state as anomalous.
                        # labels of transitions.
                        transitions_labels = []
                        # time in state.
                        time_in_state = 0
                        # number of state
                        state_idx = 0
                        # labels of the packets in the state.
                        packet_labels_in_state = []
                        for pkt_idx in range(len(anomalous_data)):
                            time_in_state += test_df.iloc[pkt_idx, 0]
                            if time_in_state == test_df.iloc[state_idx, 0]:
                                time_in_state = 0
                                state_idx += 1
                                transitions_labels.append(max(packet_labels_in_state))
                                packet_labels_in_state = []
                            else:
                                packet_labels_in_state.append(labels[pkt_idx])

                        p_x_test = test_sets_base_folder + '\\DFA\\X_test_{}_{}_{}_{}'.format(injection_length,
                                                                                              step_over,
                                                                                              percentage, epsilon)
                        p_labels = test_sets_base_folder + '\\DFA\\labels_{}_{}_{}_{}'.format(injection_length,
                                                                                              step_over,
                                                                                              percentage, epsilon)

                        Path(p_x_test).mkdir(parents=True, exist_ok=True)
                        Path(p_labels).mkdir(parents=True, exist_ok=True)

                        with open(p_x_test, mode='wb') as test_path:
                            pickle.dump(test_df, test_path)
                        with open(p_labels, mode='wb') as p_labels:
                            pickle.dump(transitions_labels, p_labels)


def create_test_input_TIRP_files_for_KL(raw_test_data_df, injection_config, input_creation_config):
    """
    make test data sets for KL.
    for each input creation option: create TIRP with all possible injection options.
    """
    pkt_df = data.load(data.datasets_path, "modbus")
    IP = data.plc
    # consider only response packets from the PLC.
    plc_df = pkt_df.loc[(pkt_df['src_ip'] == IP) & (pkt_df['dst_ip'] == IP)]
    stats_dict = data.get_plcs_values_statistics(plc_df, 5, to_df=False)

    name_2_func = {'EqualFreq': TIRP.equal_frequency_discretization, 'EqualWidth': TIRP.equal_width_discretization,
                   'KMeans': TIRP.k_means_binning}
    func_2_name = {TIRP.k_means_binning: 'kmeans', TIRP.equal_frequency_discretization: 'equal_frequency',
               TIRP.equal_width_discretization: 'equal_width'}
    # first, grid over injection params.
    with open(injection_config, mode='r') as anomalies_config:
        injection_params = yaml.load(anomalies_config, Loader=yaml.FullLoader)
        injection_lengths = injection_params['InjectionLength']
        step_overs = injection_params['StepOver']
        percentages = injection_params['Percentage']
        epsilons = injection_params['Epsilon']
        with open(input_creation_config, mode='r') as TIRP_creation_config:
            TIRP_params = yaml.load(TIRP_creation_config, Loader=yaml.FullLoader)
            binning = TIRP_params['binning']
            bins = TIRP_params['number_of_bins']
            window_params = TIRP_params['window_sizes']
            window_min = window_params['start']
            window_max = window_params['end']
            window_step = window_params['step']
            window_sizes = range(window_min, window_max, window_step)
            for injection_length in injection_lengths:
                for step_over in step_overs:
                    for percentage in percentages:
                        for epsilon in epsilons:
                            # inject in each possible way.
                            # labels will be used for creation of expected labels for the RF.
                            test_data = data.load(data.datasets_path, raw_test_data_df)
                            anomalous_data, labels = inject_to_raw_data(test_data, injection_length, step_over,
                                                                        percentage,
                                                                        epsilon)
                            for method in binning:
                                for number_of_bins in bins:
                                    for window_size in window_sizes:
                                        # discover TIRPs in separate windows.
                                        test_path_sliding_windows = test_sets_base_folder + '\\KL\\TIRP\\{}_{}_{_{}_{}_{}_{}'.format(
                                            method, number_of_bins, window_size, injection_length, step_over,
                                            percentage, epsilon)

                                        # make sure dirs exists.
                                        Path(test_path_sliding_windows).mkdir(parents=True, exist_ok=True)

                                        # discover TIRPs.
                                        # pass symbols and entities that were previously found.
                                        suffix = '\\{}_{}_{}'.format(func_2_name[name_2_func[method]], number_of_bins, window_size)
                                        symbols_path = TIRP.input.KL_symbols + suffix
                                        entities_path = TIRP.input.KL_entities + suffix
                                        with open(symbols_path, mode='rb') as syms_path:
                                            ready_symbols = pickle.load(syms_path)
                                        with open(entities_path, mode='rb') as ent_path:
                                            ready_entities = pickle.load(ent_path)
                                        TIRP.make_input(anomalous_data, name_2_func[method], number_of_bins,
                                                        window_size, consider_last=True, stats_dict=stats_dict,
                                                        test_path=test_path_sliding_windows, ready_symbols=ready_symbols, ready_entities=ready_entities)

                                        # create the labels for the RF classifier.
                                        # for each window: [start, end]
                                        test_labels_RF = []
                                        for i in range(len(anomalous_data) - window_size):
                                            # get the labels for the windows' packets.
                                            window_labels = labels[i, i + window_size]
                                            # the label is 0 for a benign packet and 1 for an anomalous packets.
                                            # so a set of packets has an anomaly in it iff the max of its corresponding labels is 1.
                                            window_label = max(window_labels)
                                            # add label.
                                            test_labels_RF.append(window_label)
                                        path = test_sets_base_folder + '\\KL\\RF\\test_labels\\{}_{}_{}_{}_{}_{}_{}'.format(
                                            method,
                                            number_of_bins,
                                            window_size,
                                            injection_length,
                                            step_over,
                                            percentage,
                                            epsilon)

                                        Path(path).mkdir(parents=True, exist_ok=True)
                                        with open(path, mode='wb') as labels_path:
                                            pickle.dump(test_labels_RF, labels_path)


def create_test_df_for_KL_based_RF(KL_config_path, injections_config_path):
    # call parse_output with the respective folders and save the dataset (without the anomaly column).
    base = data.datasets_path + "\\KL\\"
    with open(KL_config_path, mode='r') as KL_params_path:
        KL_params = yaml.load(KL_params_path, Loader=yaml.FullLoader)['KarmaLegoParams']
        binning_methods = KL_params['BinningMethods']
        bins = KL_params['Bins']
        windows = KL_params['Windows']
        epsilons = KL_params['Epsilons']
        max_gaps = KL_params['MaxGaps']
        min_ver_sups = KL_params['MinVerSups']
        # go over all whole_tirp files (defining the binning, number of bins and window size)
        for binning_method in binning_methods:
            for b in bins:
                for window in windows:
                    whole_TIRP_file_out_folder = base + 'whole_out\\all_{}_bins_{}_window_{}'.format(binning_method, b,
                                                                                                     window)
                    # iterate over all KL params, get the whole TIRPs file path.
                    for epsilon in epsilons:
                        for max_gap in max_gaps:
                            for min_ver_sup in min_ver_sups:
                                whole_TIRP_file_out = whole_TIRP_file_out_folder + "\\" + 'eps_{}_minVS_{}_maxGap_{}_HS_true.txt'.format(
                                    epsilon, min_ver_sup, max_gap)
                                with open(injections_config_path, mode='r') as injection_config:
                                    injection_params = yaml.load(injection_config, Loader=yaml.FullLoader)
                                    injection_lengths = injection_params['InjectionLength']
                                    step_overs = injection_params['StepOver']
                                    percentages = injection_params['Percentage']
                                    injection_epsilons = injection_params['Epsilon']
                                    # go over all injection params and find the TIRPs found in the test data which was made anomalous by injecting
                                    # anomalies using the injection params.
                                    for injection_length in injection_lengths:
                                        for step_over in step_overs:
                                            for percentage in percentages:
                                                for injection_epsilon in injection_epsilons:
                                                    # outDir + String.Format("\\{0}_{1}_{2}_{3}\
                                                    output_base = test_sets_base_folder + '\\KL\\KL out'
                                                    desc = '\\{}_{}_{}_{}_{}_{}_{}\\{}_{}_{}_true'.format(
                                                        binning_method, b, window, injection_length, step_over,
                                                        percentage, injection_epsilon, epsilon, min_ver_sup, max_gap)
                                                    out_dir = output_base + desc
                                                    # call parse_output
                                                    windows_TIRPs_df = TIRP.output.parse_output(whole_TIRP_file_out,
                                                                                                out_dir)
                                                    # the true labels were saved earlier.
                                                    windows_TIRPs_df_unlabeled = windows_TIRPs_df.drop(
                                                        columns=['anomaly'])
                                                    path = test_sets_base_folder + '\\KL\\RF\\test_samples\\{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_true'.format(
                                                        binning_method,
                                                        b,
                                                        window,
                                                        injection_length,
                                                        step_over,
                                                        percentage,
                                                        injection_epsilon,
                                                        epsilon, min_ver_sup, max_gap)
                                                    Path(path).mkdir(parents=True, exist_ok=True)
                                                    with open(path, mode='wb') as samples_path:
                                                        pickle.dump(windows_TIRPs_df_unlabeled, samples_path)


def make_best(results_df):
    # df for best scores without hyper-parameters being taken into consideration.
    best_df = pd.DataFrame(
        columns=['data version', 'binning', '# bins', 'precision', 'recall', 'auc', 'f1', 'injection length',
                 'step over', 'percentage', 'injection epsilon'])
    # group by the non-hyper-parameters and get the best results.
    grouped_results = results_df.groupby(by=['data version', 'binning', '# bins', 'injection length',
                                             'step over', 'percentage', 'injection epsilon'])

    for group_name, group in grouped_results:
        best_precision = max(group['precision'])
        best_recall = max(group['recall'])
        best_auc = max(group['auc'])
        best_f1 = max(group['f1'])

        best_result = {'data version': group_name[0], 'binning': group_name[1], '# bins': group_name[2],
                       'precision': best_precision, 'recall': best_recall, 'auc': best_auc, 'f1': best_f1,
                       'injection length': group_name[3],
                       'step over': group_name[4], 'percentage': group_name[5], 'injection epsilon': group_name[6]}
        temp_df = pd.DataFrame.from_dict(data={'0': best_result}, orient='index', columns=best_df.columns)
        best_df = pd.concat([best_df, temp_df])
    return best_df


def test_LSTM_based_RF(models_train_config, injection_config, tests_config_path):
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
                                        for percentage in percentages:
                                            for epsilon in epsilons:
                                                p_x_test = test_sets_base_folder + '\\LSTM_RF_OCSVM\\{}}\\X_test_{}_{}_{}_{}_{}'.format(
                                                    injection_sets_folder, desc, injection_length,
                                                    step_over, percentage, epsilon)

                                                p_y_test = test_sets_base_folder + '\\LSTM_RF_OCSVM\\{}}\\y_test_{}_{}_{}_{}_{}'.format(
                                                    injection_sets_folder, desc, injection_length,
                                                    step_over, percentage, epsilon)

                                                p_labels = test_sets_base_folder + '\\LSTM_RF_OCSVM\\{}}\\labels_{}_{}_{}_{}_{}'.format(
                                                    injection_sets_folder, desc, injection_length,
                                                    step_over, percentage, epsilon)

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
                                                          'injection epsilon': epsilon,
                                                          'percentage': percentage}
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
                                                else:
                                                    OCSVM_params_for_excel = split_model.split(sep='_')
                                                    nu = OCSVM_params_for_excel[1]
                                                    kernel = OCSVM_params_for_excel[3].split(sep='.')[0]
                                                    result['kernel'] = kernel
                                                    result['nu'] = nu
                                                    for col_name in excel_cols.difference(OCSVM_cols):
                                                        result[col_name] = '-'
                                                results_df = pd.concat([results_df,
                                                                        pd.DataFrame.from_dict(data={'0': result},
                                                                                               orient='index',
                                                                                               columns=excel_cols)],
                                                                       axis=0,
                                                                       ignore_index=True)
                                                with open(test_LSTM_RF_OCSVM_log, mode='a') as test_log:
                                                    test_log.write('recorded results of model from type: {}\n'.format(prefix + model_type))
                                                    test_log.write('injection parameters are: len: {}, step: {}, %: {}, eps: {}\n'.format(injection_length, step_over, percentage, epsilon))
                                                    test_log.write('model parameters:\n')
                                                    test_log.write('data version: {}, # bins: {}, binning method: {}\n'.format(data_version_for_excel, number_of_bins_for_excel, binning_method_for_excel))
                                                    if model_type == 'RF':
                                                        test_log.write('# estimators: {}, criterion: {}, max features: {}\n'.format(result['# estimators'], result['criterion'], result['max features']))
                                                    else:
                                                        test_log.write('kernel: {}, nu: {}\n'.format(result['kernel'], result['nu']))
                                                    test_log.write('scores: precision: {}, recall: {}, auc: {}, f1: {}\n'.format(result['precision'], result['recall'], result['auc scores'], result['f1']))
            best_df = make_best(results_df)

            with pd.ExcelWriter(xl_path) as writer:
                best_df['name'] = prefix + model_type
                results_df['name'] = prefix + model_type
                sheet = prefix + model_type + ' performance'
                results_df.to_excel(excel_writer=writer, sheet_name=sheet)
                sheet = prefix + model_type + ' best scores'
                best_df.to_excel(excel_writer=writer, sheet_name=sheet)


def test_DFA(injection_config):
    with open(data.automaton_path + '\\DFA', mode='rb') as DFA_p:
        DFA = pickle.load(DFA_p)
    with open(injection_config, mode='r') as injections:
        injection_params = yaml.load(injections, Loader=yaml.FullLoader)
    injection_lengths = injection_params['InjectionLength']
    step_overs = injection_params['StepOver']
    percentages = injection_params['Percentage']
    epsilons = injection_params['Epsilon']
    results_df = pd.DataFrame(columns=excel_cols)
    for injection_length in injection_lengths:
        for step_over in step_overs:
            for percentage in percentages:
                for epsilon in epsilons:
                    p_x_test = test_sets_base_folder + '\\DFA\\X_test_{}_{}_{}_{}'.format(injection_length, step_over,
                                                                                          percentage, epsilon)
                    p_labels = test_sets_base_folder + '\\DFA\\labels_{}_{}_{}_{}'.format(injection_length, step_over,
                                                                                          percentage, epsilon)
                    with open(p_x_test, mode='rb') as test_data_path:
                        test_df = pickle.load(test_data_path)
                    with open(p_labels, mode='rb') as labels_path:
                        test_labels = pickle.load(labels_path)

                    # the DFA classifies transitions.
                    decisions = models.automaton.detect(DFA, test_df, DFA_regs)

                    precision, recalls, thresholds = precision_recall_curve(
                        y_true=test_labels,
                        probas_pred=decisions)
                    precision = precision[0]
                    recall = recalls[0]
                    auc_score = roc_auc_score(y_true=test_labels, y_score=decisions)
                    f1 = f1_score(y_true=test_labels, y_pred=decisions)

                    result = {'injection length': injection_length,
                              'step over': step_over,
                              'injection epsilon': epsilon,
                              'percentage': percentage,
                              'precision': precision,
                              'recall': recall,
                              'auc': auc_score,
                              'f1': f1}
                    for col_name in excel_cols.difference(DFA_cols):
                        result[col_name] = '-'
                    results_df = pd.concat(
                        [results_df, pd.DataFrame.from_dict(data={'0': result}, columns=excel_cols, orient='index')],
                        axis=0, ignore_index=True)
                    with open(test_DFA_log, mode='a') as test_log:
                        test_log.write('recorded DFA results for injection with parameters:\n')
                        test_log.write('len: {}, step: {}, %: {}, eps: {}\n'.format(injection_length, step_over, percentage, epsilon))
                        test_log.write(
                            'scores: precision: {}, recall: {}, auc: {}, f1: {}\n'.format(result['precision'], result['recall'], result['auc scores'], result['f1']))

    # this already writes the best results achieved by DFA because there is only 1 version of it.
    with pd.ExcelWriter(xl_path) as writer:
        results_df['name'] = 'DFA'
        results_df.to_excel(excel_writer=writer, sheet_name='DFA performance')


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
                                        whole_TIRP_file_out_folder = input_base + "whole_out\\all_" + binning_method + "_bins_{0}_window_{1}".format(
                                            number_of_bins, window)
                                        all_TIRPs_path = whole_TIRP_file_out_folder + "\\" + "eps_{0}_minVS_{1}_maxGap_{2}_HS_true.txt".format(
                                            KL_epsilon, min_ver_sup, max_gap)
                                        for injection_length in injection_lengths:
                                            for step_over in step_overs:
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

                                                        test_df = models.TIRP.output.parse_output(all_TIRPs_path,
                                                                                                  windows_outputs_dir)

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
                                                                  'injection epsilon': injection_epsilon,
                                                                  'percentage': percentage,
                                                                  'precision': precision,
                                                                  'recall': recall,
                                                                  'auc': auc_score,
                                                                  'f1': f1}
                                                        temp_df = pd.DataFrame.from_dict(columns=excel_cols,
                                                                                         data={'0': result},
                                                                                         orient='index')
                                                        results_df = pd.concat([results_df, temp_df], axis=0,
                                                                               ignore_index=True)
                                                        with open(test_KL_RF_log, mode='a') as test_log:
                                                            test_log.write('recorded results for KL-RF for injection with parameters:\n')
                                                            test_log.write('len: {}, step: {}, %: {}, eps: {}\n'.format(injection_length, step_over, percentage, injection_epsilon))
                                                            test_log.write('model parameters:\n')
                                                            test_log.write('binning: {}, # bins: {}, window size: {}, KL epsilon: {}, minimal VS: {}, max gap: {}\n'.format(binning_method, number_of_bins, window,
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


if __name__ == '__main__':
    with open('C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\config\\test sets processing config.yaml', mode='r') as path:
        file = yaml.load(path, Loader=yaml.FullLoader)
    proc = file['processing_config']
    for k in proc:
        print(k)

