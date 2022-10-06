import unittest

import automaton

states = [{'30': 10, '15': 20, '120': 45},
          {'30': 1, '15': 2, '120': 43}]
registers = ['15', '30', '120']
same = {'30': 1, '15': 20, '120': 45}


class TestDFA(unittest.TestCase):
    def test_new_state_not_new(self):
        state = {'30': 10, '15': 20, '120': 45}
        assert automaton.new_state(state, states, registers) is False
        state = {'30': 1, '15': 20, '120': 45}
        assert automaton.new_state(state, states, registers) is True

    def test_new_state_new(self):
        state = {'30': 1, '15': 20, '120': 45}
        assert automaton.new_state(state, states, registers) is True

    def test_state_number_diffs(self):
        state = {'30': 1, '15': 20, '120': 45}
        assert automaton.state_number_of_diffs(state, same) == 0
        assert automaton.state_number_of_diffs(state, states[0]) == 1
        assert automaton.state_number_of_diffs(state, states[1]) == 2

    def test_state_difference(self):
        assert automaton.state_difference(same, same) == dict()
        diff1 = {'30': 10}
        diff2 = {'15': 2, '120': 43}
        assert automaton.state_difference(same, states[0]) == diff1
        assert automaton.state_difference(same, states[1]) == diff2

    def test_new_transition(self):
        t1 = automaton.Transition({'30': 1, '15': 2, '120': 3}, {'30': 4, '15': 5, '120': 6}, {'30': 4, '15': 5, '120': 6}, 0)
        t2 = automaton.Transition({'30': 11, '15': 21, '120': 31}, {'30': 11, '15': 51, '120': 61}, {'15': 51, '120': 61}, 1)
        t3 = automaton.Transition({'30': 23, '15': 24, '120': 1}, {'30': 23, '15': 24, '120': 61}, {'120': 61}, 2)
        t4 = automaton.Transition(states[0], states[1], {'30': 1, '15': 2, '120': 43}, 3)
        old_ts = [t1, t2, t3]
        assert automaton.new_transition(t4, old_ts) is True
        old_ts.append(t4)
        assert automaton.new_transition(t4, old_ts) is False

    def test_find_same_found(self):
        t1 = automaton.Transition({'30': 1, '15': 2, '120': 3}, {'30': 4, '15': 5, '120': 6},
                                  {'30': 4, '15': 5, '120': 6}, 0)
        t2 = automaton.Transition({'30': 11, '15': 21, '120': 31}, {'30': 11, '15': 51, '120': 61},
                                  {'15': 51, '120': 61}, 1)
        t3 = automaton.Transition({'30': 23, '15': 24, '120': 1}, {'30': 23, '15': 24, '120': 61}, {'120': 61}, 2)
        t4 = automaton.Transition(states[0], states[1], {'30': 1, '15': 2, '120': 43}, 3)
        old_ts = [t1, t2, t3]
        assert automaton.find_same(t4, old_ts) is None
        old_ts.append(t4)
        res = automaton.find_same(t4, old_ts)
        assert automaton.state_difference(res.final_state, t4.final_state) == dict() and automaton.state_difference(t4.initial_state, res.initial_state) == dict()


