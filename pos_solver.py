###################################
# CS B551 Fall 2016, Assignment #3
#
# Your names and user ids:
#   Sarvothaman Madhavan    -   madhavas
#   Raghavendra Nataraj     -   natarajr
#   Prateek Srivastava      -   pratsriv
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Training:
# While training the following probabilities are calculated using the plug-in principle (plug in the counts/total
# occurrence in place of probabilities:
# 1. Initial state probabilities            :   Out of all the sentences, how many times did each part of speech
#                                               started the sentence
# 2. State transition probabilities         :   Count the occurrence of each pairs of pos that occurred in training data
# 3. Emission probabilities                 :   For each pos, count how many times any word occurred
#                                               as that part of speech
# 4. Complex state transition probabilities :   Count the occurrence of each triple pairs of pos that occurred
#                                               in training data.
#
#
# Complex Model:
# For the complex model, we started out with estimating the P(S1/W) and P(S2/W) as special cases since these do not have
# the same structure as all other probabilities i.e P(S3/W). Once these two are saved as tau1 and tau2, all other
# probability calculations will lookup previous tau values to estimate current "level" probabilities and further save
# it as current level of tau
#
# Posterior Calculation:
#           For posterior calculation, we assume the HMM model
#
# Accuracy Table for bc.test
# --------------------------------------------------------
#               |   Word Accuracy   |   Sentence Accuracy
# --------------------------------------------------------
# Simplified    |       93.96%      |       47.50%
# --------------------------------------------------------
# hmm           |       95.03%      |       54.05%
# --------------------------------------------------------
# Complex       |       92.61%      |       44.45%
# --------------------------------------------------------

####


import math
import operator
import itertools


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
    taus = {}
    emission_cost = {}
    pos_transition_cost = {}
    min_emission = 0.0000001
    min_transition = 0.0000001
    min_complex_transition = 0.0000001

    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        neg_log = 0
        previous_pos=None
        for word, pos in zip(sentence, label):
            emission_cost = math.log(self.min_emission)
            if word in self.emission_cost[pos]:
                emission_cost = math.log(self.emission_probabilities[pos][word])
            transition_cost= math.log(self.min_transition)
            if previous_pos is None:
                transition_cost=math.log(self.pos_init_probabilities[pos])
            else:
                if pos in self.pos_transition_cost[previous_pos]:
                    transition_cost = math.log(self.pos_transition_probabilities[previous_pos][pos])
            neg_log += emission_cost + transition_cost
            previous_pos=pos
        return neg_log

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

        for word_list, pos_list in data:
            trigram_pos = None
            previous_pos = None
            first_state_flag = 1
            for pos, word in zip(pos_list, word_list):
                word_count += 1
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
        to_add = prior_pos.difference(set(pos_init_counts.keys()))

        # Add a very small value for count if any pos is missing in prior
        for item in to_add:
            pos_init_counts[item] = 0.01

        # Convert initial state counts to probabilities
        for pos, count in pos_init_counts.items():
            self.pos_init_probabilities[pos] = (1.0 * count) / sentence_count

        prior_pos_copy = prior_pos.copy()
        for previous_pos, pos_dict in pos_transition_counts.items():
            prior_pos_copy.remove(previous_pos)
            strike_off = prior_pos.copy()
            for current_pos, count in pos_dict.items():
                strike_off.remove(current_pos)

            # Add a very small value for count if any pos is missing in prior
            for inner_item in strike_off:
                pos_transition_counts[previous_pos][inner_item] = 0.01

        # Add a very small value for count if any pos is missing in prior
        for item in prior_pos_copy:
            pos_transition_counts[item] = {}
            for inner_item in prior_pos:
                pos_transition_counts[item][inner_item] = 0.01

        pos_pairs = []
        for i in itertools.product(prior_pos, prior_pos):
            pos_pairs.append(i)
        for previous_pos_pair, pos_dict in pos_complex_transition_counts.items():
            pos_pairs.remove(previous_pos_pair)
            strike_off = prior_pos.copy()
            for current_pos, count in pos_dict.items():
                strike_off.remove(current_pos)
            for inner_item in strike_off:
                pos_complex_transition_counts[previous_pos_pair][inner_item] = 0.01
        for item in pos_pairs:
            pos_complex_transition_counts[item] = {}
            for inner_item in prior_pos:
                pos_complex_transition_counts[item][inner_item] = 0.01

        # Convert state transitions counts to probabilities
        for previous_pos, pos_dict in pos_transition_counts.items():
            current_count = sum(pos_dict.values())
            self.pos_transition_probabilities[previous_pos] = {}
            self.pos_transition_cost[previous_pos] = {}
            for current_pos, count in pos_dict.items():
                self.pos_transition_probabilities[previous_pos][current_pos] = ((
                                                                                    1.0 * count) / current_count) + 0.0000001
                self.pos_transition_cost[previous_pos][current_pos] = math.log(
                    1.0 / self.pos_transition_probabilities[previous_pos][current_pos])

        # Convert emission counts to probabilities
        emission_sanity_count = 0
        for pos, word_dict in emission_counts.items():
            current_count = sum(word_dict.values())
            emission_sanity_count += current_count
            self.emission_probabilities[pos] = {}
            self.emission_cost[pos] = {}
            for word, count in word_dict.items():
                self.emission_probabilities[pos][word] = ((1.0 * count) / current_count) + 0.0000001
                self.emission_cost[pos][word] = math.log(1.0 / self.emission_probabilities[pos][word])

        # Convert state transition counts to probabilities for complex models
        complex_transitions_sanity_count = 0
        for pos_combination, pos_dict in pos_complex_transition_counts.items():
            current_count = sum(pos_dict.values())
            complex_transitions_sanity_count += current_count
            self.pos_complex_transition_probabilities[pos_combination] = {}
            for current_pos, count in pos_dict.items():
                self.pos_complex_transition_probabilities[pos_combination][current_pos] = ((
                                                                                               1.0 * count) / current_count) + 0.0000001

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        pos_sent = []
        pro_sent = []
        for word in sentence:
            max_pro = 0;
            max_pos = 0;
            base = 0
            for idx, pos_type in enumerate(self.prior.keys()):
                if word in self.emission_probabilities[pos_type]:
                    pr = self.emission_probabilities[pos_type][word] * self.prior[pos_type]
                else:
                    pr = 0.0000001 * self.prior[pos_type]
                base += pr
                if pr > max_pro:
                    max_pos = idx
                    max_pro = pr
            pos_sent.append(self.prior.keys()[max_pos])
            pro_sent.append(round(max_pro / base, 4))
        return [[pos_sent], [pro_sent]]

    def hmm(self, sentence):
        veterbi = {0: {}}
        # veterbi[0] = {}
        for idx, pos_type in enumerate(self.prior.keys()):
            word = sentence[0]
            if sentence[0] in self.emission_cost[pos_type]:
                veterbi[0][pos_type] = (
                    [], math.log(1 / self.pos_init_probabilities[pos_type]) + self.emission_cost[pos_type][word])
            else:
                veterbi[0][pos_type] = (
                    [], math.log(1 / self.pos_init_probabilities[pos_type]) + math.log(1 / 0.0000001))
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
                    veterbi[index][pos_type] = (path, min(prob_values) + math.log(1 / 0.0000001))
        pos_sent = []

        for index in range(0, len(sentence)):
            max_val = min(veterbi[index].iteritems(), key=operator.itemgetter(1))[0]
            pos_sent.append(max_val)
        min_index = min(veterbi[len(sentence) - 1])
        final_pos = veterbi[len(sentence) - 1][min_index][0]
        final_pos.append(min_index)
        return [[final_pos], []]

    def complex(self, sentence):
        pos = self.prior.keys()
        pos_list = []
        conf_list = []
        for i, v in enumerate(sentence):
            prob_max = 0
            base = 0
            final_pos = pos[0]
            tau_value = "s" + str(i + 1)
            tau_lookup_value = "s" + str(i)
            if i == 0:
                for s_1 in self.pos_init_probabilities.keys():
                    emission_prob = self.min_emission
                    if v in self.emission_probabilities[s_1]:
                        emission_prob = self.emission_probabilities[s_1][v]
                    calculated_tau_value = self.pos_init_probabilities[s_1] * emission_prob
                    if tau_value in self.taus:
                        self.taus[tau_value][s_1] = calculated_tau_value
                    else:
                        self.taus[tau_value] = {}
                        self.taus[tau_value][s_1] = calculated_tau_value
                    base += calculated_tau_value

                    if calculated_tau_value > prob_max:
                        prob_max = calculated_tau_value
                        final_pos = s_1
            elif i == 1:
                for s_2 in pos:
                    curr_pos_sum = 0
                    for s_1 in pos:
                        current_tau_value = self.taus[tau_lookup_value][s_1]
                        emission_prob = self.min_emission
                        if v in self.emission_probabilities[s_2]:
                            emission_prob = self.emission_probabilities[s_2][v]

                        transition_prob = self.min_transition
                        if s_2 in self.pos_transition_probabilities[s_1]:
                            transition_prob = self.pos_transition_probabilities[s_1][s_2]

                        calculated_tau_value = current_tau_value * transition_prob * emission_prob
                        curr_pos_sum += calculated_tau_value
                        if tau_value in self.taus:
                            if s_1 in self.taus[tau_value]:
                                self.taus[tau_value][s_1][s_2] = calculated_tau_value
                            else:
                                self.taus[tau_value][s_1] = {}
                                self.taus[tau_value][s_1][s_2] = calculated_tau_value
                        else:
                            self.taus[tau_value] = {}
                            self.taus[tau_value][s_1] = {}
                            self.taus[tau_value][s_1][s_2] = calculated_tau_value
                    base += curr_pos_sum
                    if curr_pos_sum > prob_max:
                        prob_max = curr_pos_sum
                        final_pos = s_2
            else:
                for s_3 in pos:
                    curr_pos_sum = 0
                    for s_1 in pos:
                        for s_2 in pos:
                            current_tau_value = self.taus[tau_lookup_value][s_1][s_2]

                            transition_prob = self.min_complex_transition
                            if (s_1, s_2) in self.pos_complex_transition_probabilities and s_3 in \
                                    self.pos_complex_transition_probabilities[(s_1, s_2)]:
                                transition_prob = self.pos_complex_transition_probabilities[(s_1, s_2)][s_3]

                            emission_prob = self.min_emission
                            if v in self.emission_probabilities[s_3]:
                                emission_prob = self.emission_probabilities[s_3][v]

                            calculated_tau_value = current_tau_value * transition_prob * emission_prob
                            curr_pos_sum += calculated_tau_value
                            if tau_value in self.taus:
                                if s_2 in self.taus[tau_value]:
                                    self.taus[tau_value][s_2][s_3] = calculated_tau_value
                                else:
                                    self.taus[tau_value][s_2] = {}
                                    self.taus[tau_value][s_2][s_3] = calculated_tau_value
                            else:
                                self.taus[tau_value] = {}
                                self.taus[tau_value][s_2] = {}
                                self.taus[tau_value][s_2][s_3] = calculated_tau_value
                    base += curr_pos_sum
                    if curr_pos_sum > prob_max:
                        prob_max = curr_pos_sum
                        final_pos = s_3
            pos_list.append(final_pos)
            if base != 0:
                conf_list.append(round(prob_max / base, 4))
            else:
                conf_list.append(0.9999)
        return [[pos_list], [conf_list]]

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
