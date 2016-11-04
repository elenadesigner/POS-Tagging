###################################
# CS B551 Fall 2016, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
####

import random
import math
import collections
import pprint
import operator


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    prior = {}
    pos_init_probabilities = {}
    pos_transition_probabilities = {}
    pos_complex_transition_probabilities = {}
    emission_probabilities = {}

    emission_cost = {}
    pos_transition_cost = {}

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        return 0

    # Do the training!
    #
    def train(self, data):
        prior_counts = {}
        pos_init_counts = {}
        pos_transition_counts = {}
        pos_complex_transition_counts = {}
        emission_counts = {}
        word_count = 0
        sentence_count = 0
        trigram_pos = None
        previous_pos = None

        for word_list, pos_list in data:
            first_state_flag = 1
            for pos, word in zip(pos_list, word_list):
                # Calculate initial state counts
                if first_state_flag == 1:
                    sentence_count += 1
                    if pos in pos_init_counts:
                        pos_init_counts[pos] += 1
                    else:
                        pos_init_counts[pos] = 1
                    first_state_flag = 0
                else:
                    # Calculate state transition counts
                    if previous_pos in pos_transition_counts:
                        if pos in pos_transition_counts[previous_pos]:
                            pos_transition_counts[previous_pos][pos] += 1
                        else:
                            pos_transition_counts[previous_pos][pos] = 1
                    else:
                        pos_transition_counts[previous_pos] = {}
                        pos_transition_counts[previous_pos][pos] = 1

                    # Calculate transition counts for complex model
                    if trigram_pos is not None:
                        if (trigram_pos, previous_pos) in pos_complex_transition_counts:
                            if pos in pos_complex_transition_counts[(trigram_pos, previous_pos)]:
                                pos_complex_transition_counts[(trigram_pos, previous_pos)][pos] += 1
                            else:
                                pos_complex_transition_counts[(trigram_pos, previous_pos)][pos] = 1
                        else:
                            pos_complex_transition_counts[(trigram_pos, previous_pos)] = {}
                            pos_complex_transition_counts[(trigram_pos, previous_pos)][pos] = 1
                    trigram_pos = previous_pos

                # Calculate emission counts
                if pos in emission_counts:
                    if word in emission_counts[pos]:
                        emission_counts[pos][word] += 1
                    else:
                        emission_counts[pos][word] = 1
                else:
                    emission_counts[pos] = {}
                    emission_counts[pos][word] = 1
                # Calculate prior counts
                word_count += 1
                if pos in prior_counts:
                    prior_counts[pos] += 1
                else:
                    prior_counts[pos] = 1
                previous_pos = pos

        # Convert prior counts to probabilities
        for pos, count in prior_counts.items():
            self.prior[pos] = (1.0 * count) / word_count

        # List of prior value
        prior_pos = set(self.prior.keys())
        to_add = prior_pos.difference(self.pos_init_probabilities.keys())

        # Set missing pos to very small value (Laplace smoothing)
        for item in to_add:
            self.pos_init_probabilities[item] = 0.00001

        # Convert initial state counts to probabilities
        for pos, count in pos_init_counts.items():
            self.pos_init_probabilities[pos] = (1.0 * count) / sentence_count

        prior_pos_copy = prior_pos.copy()
        # Convert state transitions counts to probabilities
        for previous_pos, pos_dict in pos_transition_counts.items():
            prior_pos.remove(previous_pos)
            current_count = sum(pos_dict.values())
            self.pos_transition_probabilities[previous_pos] = {}
            self.pos_transition_cost[previous_pos] = {}
            strike_off = prior_pos_copy.copy()
            for current_pos, count in pos_dict.items():
                # Strike off pos values seen so far
                strike_off.remove(current_pos)

                self.pos_transition_probabilities[previous_pos][current_pos] = ((1.0 * count) / current_count) + 0.00001
                self.pos_transition_cost[previous_pos][current_pos] = math.log(
                    1.0 / self.pos_transition_probabilities[previous_pos][current_pos])

            # Set very small values for transitions not seen so far
            for inner_item in strike_off:
                self.pos_transition_probabilities[previous_pos][inner_item] = 0.00001
                self.pos_transition_cost[previous_pos][inner_item] = math.log(
                    1.0 / self.pos_transition_probabilities[previous_pos][inner_item])

        # Set very small values for transitions not seen so far
        for item in prior_pos:
            self.pos_transition_probabilities[item] = {}
            self.pos_transition_cost[item] = {}
            for inner_item in prior_pos_copy:
                self.pos_transition_probabilities[item][inner_item] = 0.00001
                self.pos_transition_cost[item][inner_item] = math.log(
                    1.0 / self.pos_transition_probabilities[item][inner_item])

        # Convert emission counts to probabilities
        for pos, word_dict in emission_counts.items():
            current_count = sum(word_dict.values())
            self.emission_probabilities[pos] = {}
            self.emission_cost[pos] = {}
            for word, count in word_dict.items():
                self.emission_probabilities[pos][word] = ((1.0 * count) / current_count) + 0.00001
                self.emission_cost[pos][word] = math.log(1.0 / self.emission_probabilities[pos][word])
        # Convert state transitions counts to probabilities
        for pos_combination, pos_dict in pos_complex_transition_counts.items():
            current_count = sum(pos_dict.values())
            self.pos_complex_transition_probabilities[pos_combination] = {}
            for current_pos, count in pos_dict.items():
                self.pos_complex_transition_probabilities[pos_combination][current_pos] = ((
                                                                                           1.0 * count) / current_count) + 0.00001

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        pos_sent = []
        pro_sent = []
        for word in sentence:
            max_pro = 0;
            max_pos = 0;
            for idx, pos_type in enumerate(self.prior.keys()):
                if word in self.emission_probabilities[pos_type]:
                    pr = self.emission_probabilities[pos_type][word] * self.prior[pos_type]
                else:
                    pr = 0.00001
                if (pr > max_pro):
                    max_pos = idx
                    max_pro = pr
            pos_sent.append(self.prior.keys()[max_pos])
            pro_sent.append(max_pro)
        return [[pos_sent], [pro_sent]]
        # return [ [ [ "noun" ] * len(sentence)], [[0] * len(sentence),] ]

    def hmm(self, sentence):
        veterbi = {}
        veterbi[0] = {}
        for idx, pos_type in enumerate(self.prior.keys()):
            word = sentence[0]
            if sentence[0] in self.emission_cost[pos_type]:
                veterbi[0][pos_type] = (
                    [], math.log(1 / self.pos_init_probabilities[pos_type]) + self.emission_cost[pos_type][word])
            else:
                veterbi[0][pos_type] = ([], math.log(1 / self.pos_init_probabilities[pos_type]) + math.log(1 / 0.00001))
        for index in range(1, len(sentence)):
            word = sentence[index]
            veterbi[index] = {}
            for idx, pos_type in enumerate(self.prior.keys()):
                prob_values = []
                for idx1, pos_type1 in enumerate(self.prior.keys()):
                    prob_values.append(veterbi[index - 1][pos_type1][1] + self.pos_transition_cost[pos_type1][pos_type])
                min_value = min(prob_values)
                min_pos = self.prior.keys()[prob_values.index(min_value)]
                path = veterbi[index - 1][min_pos][0][:]
                path.append(min_pos)
                if word in self.emission_cost[pos_type]:
                    veterbi[index][pos_type] = (path, min(prob_values) + self.emission_cost[pos_type][word])
                else:
                    veterbi[index][pos_type] = (path, min(prob_values) + math.log(1 / 0.00001))
                    # print veterbi[index]
        pos_sent = []
        pro_sent = []

        for index in range(0, len(sentence)):
            max_val = min(veterbi[index].iteritems(), key=operator.itemgetter(1))[0]
            pos_sent.append(max_val)
        min_index = min(veterbi[len(sentence) - 1])
        final_pos = veterbi[len(sentence) - 1][min_index][0]
        final_pos.append(min_index)
        # print final_pos
        return [[final_pos], []]

    # This will be used for the variable elimination part and will be called in complex()
    def trans_prob(self, state, ind):
        if ind - 1 > 0:
            tau = trans_prob(self, state, ind - 1)
        tau1 = tau * self.pos_transition_probabilities
        return tau

    def complex(self, sentence):
        return [[["noun"] * len(sentence)], [[0] * len(sentence), ]]

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It's supposed to return a list with two elements:
    #
    #  - The first element is a list of part-of-speech labelings of the sentence.
    #    Each of these is a list, one part of speech per word of the sentence.
    #
    #  - The second element is a list of probabilities, one per word. This is
    #    only needed for simplified() and complex() and is the marginal probability for each word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM":
            return self.hmm(sentence)
        elif algo == "Complex":
            return self.complex(sentence)
        else:
            print "Unknown algo!"
