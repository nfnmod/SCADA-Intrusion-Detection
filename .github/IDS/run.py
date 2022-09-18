import csv
import itertools
import pickle
import time
from pathlib import Path

import tensorflow
import yaml

import data
import models
import models.TIRP as TIRP

from data.injections import inject_to_raw_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

KL_base = data.datasets_path + "\\KL\\"
HTM_base = "C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\HTM\\"
logs = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\log files\\'
test_sets_base_folder = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\test sets'
KL_based_RF_log = logs + 'KarmaLego based RF.txt'
DFA_log = logs + 'DFA.txt'


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
        KL_params = params['KarmaLego_params']
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
                path_suffix = "\\eps_{0}_minVS_{1}_maxGap_{2}_HS_{3}".format(epsilon, ver_supp, max_gap, True)
                whole_TIRPS_output_path = whole_TIRPs_folder + path_suffix + ".txt"
                windows_outputs_folder_path = windows_folders_folder + path_suffix
                TIRP_df = TIRP.output.parse_output(whole_TIRPS_output_path, windows_outputs_folder_path)
                windows_features = TIRP_df[:, :-1]
                windows_labels = TIRP_df[:, -1]
                X_train, X_test, y_train, y_test = train_test_split(windows_features, windows_labels, test_size=0.2)
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
                        models_base_path = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\KL RF'
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

    binners = [data.k_means_binning, data.equal_frequency_discretization, data.equal_width_discretization]
    folder_names = {data.k_means_binning: "KMeans", data.equal_frequency_discretization: "EqualFreq",
                    data.equal_width_discretization: "EqualWidth"}
    names = {data.k_means_binning: "k_means", data.equal_frequency_discretization: "equal_frequency",
             data.equal_width_discretization: "equal_width"}
    pkt_df = data.load(data.datasets_path, "modbus")
    with open(HTM_input_creation_config, mode='r') as input_config:
        params = yaml.load(input_config, Loader=yaml.FullLoader)
        versions_dicts = params['processing_config']
        for version_dict in versions_dicts:
            n_bins = version_dict['bins']
            data_version = version_dict['name']
            use = version_dict['use']
            desc = version_dict['desc']
            if not use:
                pass
            else:
                options = itertools.product(binners, n_bins)
                for binner_bins in options:
                    binner = binner_bins[0]
                    bins = binner_bins[1]
                    processed_df = pkt_df
                    # X_train will be used to train the HTM network. X_test and sets created by injecting anomalies into X_test will be used
                    # for testing the HTM network.
                    X_train, X_test = train_test_split(processed_df, test_size=0.2, random_state=42)
                    # 1. write column names.
                    # 2. write columns data types.
                    # 3. write df to csv without the columns names.
                    folder = HTM_base + '\\datasets\\' + '{}_{}'.format(folder_names[binner], data_version)

                    train_path_str = folder + '\\' + "X_train_" + desc + "_{}_{}.csv".format(names[binner], bins)
                    test_path_str = folder + '\\' + "X_test_" + desc + "_{}_{}.csv".format(names[binner], bins)
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


def create_test_files_LSTM_RF_and_OCSVM(raw_test_data_df, data_versions_config, injection_config):
    """
    grid over injection params, for each combination : inject anomalies and then process the dataset using all methods.
    """
    test_data = data.load(data.datasets_path, raw_test_data_df)
    with open(injection_config, mode='r') as anomalies_config:
        injection_params = yaml.load(anomalies_config, Loader=yaml.FullLoader)
        injection_lengths = injection_params['injection_length']
        binnings = injection_params['binning']
        step_overs = injection_params['step_over']
        percentages = injection_params['percentage']
        epsilons = injection_params['epsilon']
        # first ,inject anomalies. and create the test set for: LSTM , RF and OCSVM.
        for injection_length in injection_lengths:
            for step_over in step_overs:
                for percentage in percentages:
                    for epsilon in epsilons:
                        anomalous_data, labels = inject_to_raw_data(test_data, injection_length, step_over, percentage,
                                                                    epsilon)
                        for folder_name, method_name in binnings:
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
                                        bins = data_version['bins']
                                        desc = data_version['desc']
                                        # processed test data frame.
                                        for number_of_bins in bins:
                                            test_df = data.process(anomalous_data, name, number_of_bins, method_name)
                                            # now create test data set for LSTM. Only need X_test and y_test.
                                            X_train, X_test, y_train, y_test = models.custom_train_test_split(test_df,
                                                                                                              20, 42, 0)
                                            # now save, X_test, y_test and the labels which will be used to obtain the y_test of the classifier.
                                            with open(
                                                    test_sets_base_folder + '\\{}_{}_{}\\X_test_{}_{}_{}_{}_{}'.format(
                                                            folder_name, name, number_of_bins, desc, injection_length,
                                                            step_over, percentage, epsilon), mode='wb') as data_path:
                                                pickle.dump(X_test, data_path)
                                            with open(
                                                    test_sets_base_folder + '\\{}_{}_{}\\y_test_{}_{}_{}_{}_{}'.format(
                                                            folder_name, name, number_of_bins, desc, injection_length,
                                                            step_over, percentage, epsilon), mode='wb') as data_path:
                                                pickle.dump(y_test, data_path)
                                            with open(
                                                    test_sets_base_folder + '\\{}_{}_{}\\labels_{}_{}_{}_{}_{}'.format(
                                                            folder_name, name, number_of_bins, desc, injection_length,
                                                            step_over, percentage, epsilon), mode='wb') as data_path:
                                                pickle.dump(labels, data_path)


def create_test_files_HTM(raw_test_data_df, data_version_config, injection_config):
    """
    grid over injection params, for each combination : inject anomalies and then process the dataset using all methods.
    """


def create_test_files_DFA(raw_test_data_df, injection_config):
    return None


def create_test_files_TIRP_KL(raw_test_data_df, injection_config):
    return None
