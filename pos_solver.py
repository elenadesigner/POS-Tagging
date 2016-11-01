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
    emission_probabilities={}
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
        emission_counts={}
        word_count = 0
        sentence_count = 0

        for word_list, pos_list in data:
            first_state_flag = 1
            for pos , word in zip(pos_list, word_list):
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

        for pos, count in prior_counts.items():
            self.prior[pos] = (1.0 * count) / word_count

        prior_pos=set(self.prior.keys())
        to_add=prior_pos.difference(self.pos_init_probabilities.keys())
        for item in to_add:
            self.pos_init_probabilities[item]=0.00001

        for pos, count in pos_init_counts.items():
            self.pos_init_probabilities[pos] = (1.0 * count) / sentence_count
        prior_pos_copy = prior_pos.copy()
        for previous_pos, pos_dict in pos_transition_counts.items():
            prior_pos.remove(previous_pos)
            current_count = sum(pos_dict.values())
            self.pos_transition_probabilities[previous_pos] = {}
            strike_off=prior_pos_copy.copy()
            for current_pos, count in pos_dict.items():
                strike_off.remove(current_pos)
                self.pos_transition_probabilities[previous_pos][current_pos] = ((1.0 * count) / current_count) + 0.00001
            for inner_item in strike_off:
                self.pos_transition_probabilities[previous_pos][inner_item] = 0.00001

        for item in prior_pos:
            self.pos_transition_probabilities[item]={}
            self.pos_transition_cost[item] = {}
            for inner_item in prior_pos_copy:
                self.pos_transition_probabilities[item][inner_item] = 0.00001
                self.pos_transition_cost[item][inner_item] = math.log(1.0/self.pos_transition_probabilities[item][inner_item])

        for pos, word_dict in emission_counts.items():
            current_count = sum(word_dict.values())
            self.emission_probabilities[pos] = {}
            self.emission_cost[pos] = {}
            for word, count in word_dict.items():
                self.emission_probabilities[pos][word] = ((1.0 * count) / current_count) + 0.00001
                self.emission_cost[pos][word] = math.log(1.0/self.emission_probabilities[pos][word])

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        print self.emission_cost['noun']['poet']
        pos_sent = []
        pro_sent = []
        for word in sentence:
            max_pro = 0;
            max_pos = 0;
            for idx,pos_type in enumerate(self.prior.keys()):
                if word in self.emission_probabilities[pos_type]:
                    pr = self.emission_probabilities[pos_type][word]*self.prior[pos_type]
                else:
                    pr = 0.00001
                if(pr>max_pro):
                    max_pos = idx
                    max_pro = pr
            pos_sent.append(self.prior.keys()[max_pos])
            pro_sent.append(max_pro)
        return [[pos_sent],[pro_sent]]
        #return [ [ [ "noun" ] * len(sentence)], [[0] * len(sentence),] ]


    def hmm(self, sentence):
        veterbi = {}
        veterbi[0] = {}
        for idx,pos_type in enumerate(self.prior.keys()):
            word = sentence[0]
            if sentence[0] in self.emission_probabilities[pos_type]:
                veterbi[0][pos_type] = self.pos_init_probabilities[pos_type] * self.emission_probabilities[pos_type][word]
            else:
                veterbi[0][pos_type] = self.pos_init_probabilities[pos_type] * 0.00001
        for index in range(1,len(sentence)):
            word = sentence[index]
            veterbi[index] = {}
            prob_values = []
            for idx,pos_type in enumerate(self.prior.keys()):
                for idx1,pos_type1 in enumerate(self.prior.keys()):
                    if pos_type not in self.pos_transition_probabilities[pos_type1]:
                        self.pos_transition_probabilities[pos_type1][pos_type] = 0.00001
                    prob_values.append(veterbi[index-1][pos_type1] * self.pos_transition_probabilities[pos_type1][pos_type])
                if word in self.emission_probabilities[pos_type]:
                    veterbi[index][pos_type] = max(prob_values) * self.emission_probabilities[pos_type][word]
                else:
                    veterbi[index][pos_type] = max(prob_values) * 0.00001
        pos_sent = []
        pro_sent = []
        for index in range(0,len(sentence)):
            max_val = max(veterbi[index].iteritems(), key=operator.itemgetter(1))[0]
            pos_sent.append(max_val)
        return [[pos_sent], []]

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
