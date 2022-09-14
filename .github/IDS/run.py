import itertools

import tensorflow
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import data
import models
import models.TIRP as TIRP

KL_base = data.datasets_path + "\\KL\\"


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


def train_automaton():
    pkts = data.load(data.datasets_path, "modbus")
    DFA = models.automaton.make_automaton(data.to_bin, pkts)
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
                    model.fit(X_train, y_train)
                    models_base_path = 'C:\\Users\\michael zaslavski\\OneDrive\\Desktop\\SCADA\\KL RF'
                    TIRP_level = "\\{}_bins_{}_window{}".format(binning, bins, window)
                    KL_level = path_suffix
                    models_folder = models_base_path + TIRP_level + KL_level
                    tensorflow.keras.models.save_model(model,
                                                       models_folder + '\\' + 'estimators_{}_'.format(
                                                           estimators) + 'criterion{}_'.format(
                                                           criterion) + 'features_{}.sav'.format(max_features))
