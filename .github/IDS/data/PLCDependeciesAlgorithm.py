import statistics

import numpy as np
import pandas as pd


def find_frequent_transitions_sequences(packets, time_window, support):
    """

    :param packets: The processed data describing MODBUS traffic. The data should be of devices of one group and the registers values should
           be binned. (use data processing version 3.1 to get this. Consider only registers which change over time.)
    :param time_window: The time window to consider when trying to find a frequent subsequent state.
    :param support: Minimal support for a sequence/state to be considered frequent.
    :return: frequent state transitions which the PLCs in the group show.
    """

    """
    steps:
        1. find frequent states.
        2. for each frequent state s1:
            for each other frequent state s2:
                if s1 can't entail s2:
                    pass
                else:
                    occurrences = 0
                    for each appearance s1' of s1:
                        last = 0
                        search for the earliest appearance s2' (since the last one used) of s2 such that time(s2')-time(s1')<=t:
                            occurrences += 1
                            last = index of s2' in the appearances series
                    get support.
                    throw or keep. 
    """

    # used for ordering transition sequences.
    def comparator(t_seq_and_idx):
        indices = t_seq_and_idx[0]
        my_length = len(indices)
        if my_length < longest:
            return indices + (0,) * (longest - my_length)
        return indices

    length = len(packets)
    # columns of registers.
    registers = list(packets.columns)[2:]
    # measure time.
    packets['timestamp'] = 0
    # appearances of states.
    states_appearances = {}

    for i in range(len(packets)):
        p = packets.iloc[i]

        # create a tuple of registers values describing the state.
        state_tuple = tuple()
        for reg in sorted(registers):
            state_tuple += ((reg, p[reg],),)

        appearances = states_appearances.get(state_tuple, [])
        appearances.append(i)
        states_appearances[state_tuple] = appearances

        # timestamp the current state.
        if i == 0:
            # first state is at time 0.
            packets.loc[0, 'timestamp'] = 0
        else:
            # any other state starts after the previous one has ended.
            packets.loc[i, 'timestamp'] = packets.loc[i - 1, 'timestamp'] + packets.loc[i - 1, 'time_in_state']

    frequent_states = []
    frequent_transitions = []
    # filter out the frequent states.
    for state, appearances in states_appearances:
        if len(appearances) / length >= support:
            frequent_states.append(state)

    # change into the form of (state,idx)
    flat_states = []
    for s in frequent_states:
        idx_list = states_appearances[s]
        for state_idx in idx_list:
            flat_states.append((state_idx, s,))

    # sort by appearances, this is just like sorting by time of state occurrence.
    flat_states = sorted(flat_states, key=lambda idx_s: idx_s[0])
    # create all possible transitions.
    transitions_times = {}
    transitions_indices = {}
    for i in range(len(flat_states) - 1):
        s = flat_states[i]
        f = flat_states[i + 1]
        s_idx = s[0]
        starting_state = s[1]
        f_idx = f[0]
        finished_state = f[1]
        time_delta = packets.loc[f_idx, 'timestamp'] - packets.loc[s_idx, 'timestamp']
        if time_delta <= time_window:
            transition = (starting_state, finished_state,)

            # add times
            prev_times = transitions_times.get(transition, [])
            prev_times.append(time_delta)
            transitions_times[transition] = prev_times

            # add indices
            prev_indices = transitions_indices.get(transition, [])
            prev_indices.append((s_idx, f_idx,))
            transitions_indices[transition] = prev_indices

    # discover frequent transitions of length 1.
    for transition in transitions_times.keys():
        transition_time_spaces = transitions_times[transition]
        t_supp = len(transition_time_spaces) / length  # number of occurrences of the transition.

        if t_supp >= support:
            frequent_transitions.append(transition)

    # now we want to extend the transitions into longer sequences which are frequent
    # and also occur within a time window of at most time_window time units.

    # longest sequence length. used for sorting.
    longest = 0

    fully_extended = []
    fully_extended_times = {}
    fully_extended_indices = {}

    prev_times = transitions_times
    prev_indices = transitions_indices

    # extend as long as possible.
    while len(frequent_transitions) > 1:
        flat_transitions = []
        times = prev_times
        indices = prev_indices
        components = {}

        # change into form of (transitions states, indices) = ((s1->s2->..s_n), (idx1,idx2...idx_n))
        for frequent_transition in frequent_transitions:
            # number of transitioned states.
            transition_sequence_length = len(frequent_transition)
            longest = max(longest, transition_sequence_length)

            t_seq_idx = transitions_indices[frequent_transition]
            flat_t_seq = (t_seq_idx, frequent_transition,)
            flat_transitions.append(flat_t_seq)

        flat_transitions = sorted(flat_transitions, key=lambda x: comparator(x))
        # from the sorting function we know that the transitions appear in increasing order
        # of their occurrence time and that shorter sequences are favored by the comparator.
        for i in range(len(flat_transitions) - 1):
            start = flat_transitions[i]
            start_indices = start[0]
            start_states = start[1]

            last_state = start_states[-1]
            last_idx = start_indices[-1]

            first_state_idx = start_indices[0]
            start_time = packets.loc[first_state_idx, 'timestamp']

            # we need to check every sequence which occurred after start.
            # if we extended, then break, remove them both from the candidates for the current iteration.
            # save the extended one aside for the next iteration.
            # if not, save start aside, remove from the candidates for the next iterations and continue.
            for j in range(i + 1, len(flat_transitions)):
                possible_end = flat_transitions[j]
                end_indices = possible_end[0]
                end_states = possible_end[1]

                # we need start to end with the first state of possible_end
                first_state = end_states[0]
                first_idx = end_indices[0]

                last_state_idx = end_indices[-1]
                end_time = packets.loc[last_state_idx, 'timestamp']
                time_delta = end_time - start_time
                if first_state != last_state or first_idx != last_idx or time_delta > time_window:
                    continue
                else:
                    # we can append them.
                    # extended sequence.
                    start_part = start_states
                    end_part = end_states[1:]
                    extended_transition = start_part + end_part

                    # extended indices.
                    start_part_indices = start_indices
                    end_part_indices = end_indices[1:]
                    extended_transition_indices = start_part_indices + end_part_indices

                    # add times
                    known_times = times.get(extended_transition, [])
                    known_times.append(time_delta)
                    times[extended_transition] = known_times

                    # add indices
                    known_indices = indices.get(extended_transition, [])
                    known_indices.append(extended_transition_indices)
                    indices[extended_transition] = known_indices

                    # save the smaller sequences which were extended.
                    # this will be used in case the extension isn't frequent.
                    if components.get(extended_transition, None) is None:
                        components[extended_transition] = [start, end_part]

        keep = {t: False for t in frequent_transitions}

        for i in range(len(flat_transitions) - 1):
            # now we filter out the infrequent/fully extended sequences.
            indices_and_transition = flat_transitions[i]

            transition = indices_and_transition[1]

            for j in range(i + 1, len(flat_transitions) - 1):
                following_transition_and_indices = flat_transitions[j]

                following_transition = following_transition_and_indices[1]

                extension_states = transition + following_transition[1:]

                parts = components.get(extension_states, [])
                if not parts:
                    continue
                else:
                    # this is an actual extension.
                    time_spaces = times[extension_states]
                    occurrences = len(time_spaces)
                    t_supp = occurrences / length

                    if t_supp >= support:
                        keep[transition] = True
                        keep[following_transition] = True
                    # this will be performed when the transition wasn't extended by any other one.

        for i in range(len(frequent_transitions)):
            transition = frequent_transitions[i]
            state = transition[1]
            # wasn't extended by any other transition.
            # it's fully extended.
            if not keep[state]:
                # save result and remove.
                fully_extended.append(state)
                fully_extended_times[state] = times[state]
                fully_extended_indices[state] = indices[state]
                frequent_transitions.remove(transition)

        # save times and indices for next iteration.
        prev_times = times
        prev_indices = indices

    # when getting here we have:
    # 1. longest frequent transitions (s1,s2..,s_n)
    # 2. indices of each states in each occurrence of the transition (i1,i2...,i_n)
    # 3. time deltas which are the time passed from the occurrence of s1 until the occurrence of s_n (depending on the indices)
    # we want the transitions to be ordered by the times. So, we need to flatten and sort with the indices' comparator.
    flat_transitions = []
    for frequent_transition in frequent_transitions:
        t_seq_idx = transitions_indices[frequent_transition]
        flat_t_seq = (t_seq_idx, frequent_transition,)
        flat_transitions.append(flat_t_seq)

    flat_transitions = sorted(flat_transitions, key=lambda x: comparator(x))

    return flat_transitions, prev_times, prev_indices, longest


def extract_features_v1(flat_transitions, prev_times, prev_indices, longest, registers):
    """
    :param longest: length of the longest frequent sequence of transitions.
    :param prev_indices: indices of transition sequences and of their subsequences.
    :param prev_times: same structure for times between states.
    :param flat_transitions: tuples of the form transition x state_indices.
    :param registers: the names of all the registers in the group which the transitions occur in.
    :return: a dataframe with features describing the frequent transitions.
    """

    """
    given a flat transition (s1,...sn),(i1,...in):
        break into (s_j, s_j+1) (i_j, i_j+1)
        get registers values for previous state, source state and the destination state.
        get times between the transitions.
        index of transition in the sequence.
        mark the changed registers which triggered the transition. 
    """

    return None


def extract_features_v2(flat_transitions, prev_times, prev_indices, longest, registers):
    """
    :param longest: length of the longest frequent sequence of transitions.
    :param prev_indices: indices of transition sequences and of their subsequences.
    :param prev_times: same structure for times between states.
    :param flat_transitions: tuples of the form transition x state_indices.
    :param registers: the names of all the registers in the group which the transitions occur in.
    :return: a dataframe with features describing the frequent transitions.
    """

    """
    given a flat transition (s1,...,sn), (i1,...in):
        use longest x registers features to describe each state in the sequence.
        use longest features to describe the times between states.
        sequences with length less than longest will be filled with 0 states and 0 time transitions.
    """
    return None
