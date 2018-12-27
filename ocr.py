#!/usr/bin/env python3
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# Authors: (Amit Makashir, Rohit Bapat, Akshay Rathi)
# (based on skeleton code by D. Crandall, Oct 2018)
#


'''
To execute this program use the following:
./ocr.py courier-train.png train-text.txt test-image-file.png

I have used the training data that was provided for part1, but kept a copy of it inside this directory. If there's any
error related to not find the correct path for this file, please try keeping the file inside the same directory as this
program "ocr.py". Al


***Assignment Report***

In this problem, our objective was to maximize the posterior probability:

P(l1,l2,l3.. | O1,O2,O3..)

where,  l1 = letter at index 1 which we want to recognize (hidden variable)
        O1 = subimage at index 1 (observed variable)


We have considered two classifiers for this problem:

1. Simple:

    In this approach, we make the Naive Bayes assumption that each character (hidden variable) is independent of
    any other character in that sentence. This simplifies our posterior probability and we only have to calculate the
    following probability for every index:

    P(li | Oi) = P(Oi | li) * P(li)

    where, P(Oi | li) = Emission probability
            P(li) =  Prior probability

    For calculating Emission probabilities, we compare every pixel from Oi and li and check if they match. But,
    two pixels that matched can be filled ("*") or empty (" "). We keep 4 different counts of matched and unmatched
    pixels as follows:
        i.   Matched and filled
        ii.  Matched and empty
        iii. Not matched and filled in hidden variable
        iv.  Not matched and empty in hidden variable

    As we are more interested in the pixels that have matched and are filled, we give them more weights as compared
    to pixels that have matched but are empty. I wasn't sure how the unmatched pixels should be weighted, so I tried
    different combinations and finally decided to give more weight to unmatched empty pixels than unmatched filled
    pixels. We finally calculate a weighted sum for matched and unmatched pixels.

    As Prof. Crandall explained in class, for calculating Emission probabilities we can assume a Naive bayes
    classifier and calculate probability (or score, because we have weighted these things) as follows:

    Emission probability = ((1 - m)^matched) * ((m)^(unmatched))    ... where, m = (Assumed % of noisy pixels) / 100

    Priors can be calculated by counting number of times each letter has occurred in the beginning of the sentence to
    the total number of sentences.

    We can take negative log of these probabilities (to avoid problems due to very small floats) and consider them
    as respective costs. For every index of Test image, we have 72 possible choices, each with an associated cost.
    We then choose the character with lowest cost for every index and return it as our answer for this classifier.


2. Hidden Markov Model (HMM):

    In this approach, we assume a Bayes Net where each character (hidden) is dependent on the previous character.
    Therefore, for calculating posteriors we do the following:

    For each index we have 72 possibilities and for each possibility with calculate the following,

    For the first index, as we don't have Transition probabilities,
    P(l) = Emission probability * Prior for that letter

    For every other index,
    P(l) = Emission probability * max( Probability of previous letter * Transition probability )

    where,  Emission probability = same as we calculated in the Simple classifier,
            Priors = same as we calculated in the Simple classifier,
            Transition probability = probability of the current character occurring given the previous character

    Just like in the Simple classifier, we can take negative log of these probabilities (call it "cost" instead)
    and minimize at every step. Once we have these costs of every character for every index of the Test image, we
    choose the character with lowest cost at the last index and then choose the previous character that lead us there
    and so on, till we get to the first character. This algorithm is called "Viterbi decoding" and it returns the
    most probable sequence of characters for the given Test image.

    For tuning this algorithm, we had to tune parameters inside the Emission probability function as we cannot change
    transition probabilities without changing the training data. We tried tuning "m" (% noisy pixels) and other
    parameters in Emission probability to get better results.

    We noticed that for some of the test cases, this algorithm returned a number for the first index.
    This isn't usually the case in English language, so we decided to divide the Emission probability by 2 just for the
    first index. This made the Emission probabilities less dominant over the Priors and the answers now seemed
    reasonable.

Generally the Viterbi algorithm would a give better answer than the Simple classifier and so we have chosen to print
that everytime.


We got the following results for each Test case that was provided:

0. SUPREME COURT OF THE UNITED STATES
1. Certicrari to the United States Court of -ppeala ior the Sixth Circuit
2. Nos. 14-556. Argued April 28, 2015 - Decided June 26, 2015
3. Together with No. 14-562, Tanco et al. v. Haslam, Governor of
4. Tennessee, et al., also on centiorari to the same court.
5. OpinionfofIthekCourt
6. As some of the petitioners in these cases demonstrate, marriage
7. embodies a love that may endure even past death.
8. It would misunoerstand the e men and women to say th-y diur-spect
9. the idea of marriage.
10. Their plea is that they do respect it, respect it so deeply that
11. they seek to find its fulfillment for themselves.
12. Their hope is not to be condemned to live in loneliness,
13. excluded from one of civilization's oldest institutions.
14. They ask for equal dignity in the eyes of the law.
15. ThevConstitutionggrants them-that right.
16. The judgement of the Court of hppeals for the Sixth Circuit 1s reversed.
17. It is so ordered.
18. KENNEDY, J., delivered the opinion of the Court, in which
19. GINSBURG, BREYER, SOTOMAYOR, and KAGAN, JJ., joined.

'''


from PIL import Image, ImageDraw, ImageFont
import sys
import numpy as np
import heapq
import math

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25
Total = CHARACTER_WIDTH*CHARACTER_HEIGHT
low_prob = 10**(-6)


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result


def load_training_letters(fname):
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }


#####
# main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

## Below is just some sample code to show you how the functions above work. 
# You can delete them and put your own code here!

# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:

# Same with test letters. Here's what the third letter of the test data
#  looks like:
# print("\n".join([ r for r in test_letters[0] ]))


def train(data):
    letter_priors={}
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    TRAIN_LETTERS=list(TRAIN_LETTERS)

    for sentence_ind in range(len(data)):
        if data[sentence_ind][0][0][0] in TRAIN_LETTERS:
            if data[sentence_ind][0][0][0] not in letter_priors.keys():
                letter_priors[data[sentence_ind][0][0][0]]=0
            letter_priors[data[sentence_ind][0][0][0]]+=1
    total = sum(letter_priors.values(), 0.0)
    letter_priors = {k: v / total for k, v in letter_priors.items()}

    missing=set(TRAIN_LETTERS)-letter_priors.keys()
    for word in missing:
        if word in TRAIN_LETTERS:
            letter_priors[word]=0.000001

    trans_probability = {}
    for i in range(len(data)):
        for pos in range(len(data[i][1])-1):
            if data[i][1][pos] not in trans_probability.keys():
                trans_probability[data[i][1][pos]]={}
            if data[i][1][pos+1] not in trans_probability[data[i][1][pos]].keys():
                trans_probability[data[i][1][pos]][data[i][1][pos+1]]=0
            trans_probability[data[i][1][pos]][data[i][1][pos+1]]+=1

    for key_val in trans_probability.keys():
        total = sum(trans_probability[key_val].values(), 0.0)
        trans_probability[key_val] = {k: v / total for k, v in trans_probability[key_val].items()}

    for word in data[0][0]:
        str_letter=''.join(word)
    letter_trans={}

    for sentence_ind in range(len(data)):
        word_list=" ".join(data[sentence_ind][0])
        letter_list=list(word_list)
        for letter_ind in range(len(letter_list)-1):
            if letter_list[letter_ind] not in letter_trans.keys():
                letter_trans[letter_list[letter_ind]]={}
            if letter_list[letter_ind+1] not in letter_trans[letter_list[letter_ind]].keys():
                letter_trans[letter_list[letter_ind]][letter_list[letter_ind+1]]=0
            letter_trans[letter_list[letter_ind]][letter_list[letter_ind+1]]+=1

    for key_val in letter_trans.keys():
        total = sum(letter_trans[key_val].values(), 0.0)
        letter_trans[key_val] = {k: v / total for k, v in letter_trans[key_val].items()}

    return letter_priors,letter_trans


def read_data(fname):
    exemplars = []
    file = open(fname, 'r');
    for line in file:
        data = tuple([w for w in line.split()])
        exemplars += [ (data[0::2], data[1::2]), ]

    for line in file:
        data = tuple([w.upper() for w in line.split()])
        exemplars += [(data[0::2], data[1::2]), ]

    for line in file:
        data = tuple([w.lower() for w in line.split()])
        exemplars += [(data[0::2], data[1::2]), ]
    return exemplars


def matched_pixels(hidden, observed):
    ignore_width = 1
    ignore_height = 2

    filled_match = 0
    empty_match = 0
    filled_unmatch = 0
    empty_unmatch = 0

    for i in range(ignore_height, CHARACTER_HEIGHT - ignore_height):
        for j in range(ignore_width, CHARACTER_WIDTH - ignore_width):
            if hidden[i][j] == observed[i][j]:
                if hidden[i][j] == "*":
                    filled_match += 1
                else:
                    empty_match += 1
            else:
                if hidden[i][j] == "*":
                    filled_unmatch += 1
                else:
                    empty_unmatch += 1

    matched = (5 * filled_match) + (empty_match / 5.5)
    unmatched = (filled_unmatch / 4) + (1.5 * empty_unmatch)

    return matched, unmatched


def emission_probability(observed,hidden):
    m = 0.01     # Tune this as per your needs
    matched,unmatched = matched_pixels(hidden,observed)
    prob = ((1 - m) ** matched) * ((m) ** (unmatched))
    if prob == 0:
        return -math.log(low_prob)
    else:
        return -math.log(prob)


# Simplified model
def simple_model(test_letters,train_letters,initial_prob):
    '''
    Assuming that every character is independent of any other character observed
    :param test_letters: array of observed letters in ocr problem
    :param train_letters: Dictionary returned by load_training_letters function
    :return:
    '''
    total_emission = []
    for test_letter in test_letters:
        emission_for_each_loc = []
        for letter, letter_array in train_letters.items():
            # no_of_mismatched = sum(c1 != c2 for i, j in zip(letter_array, test_letter) for c1, c2 in zip(i, j))
            cost = emission_probability(test_letter,letter_array) + (-math.log(initial_prob[letter]))
            heapq.heappush(emission_for_each_loc,(cost,letter))
        total_emission.append(emission_for_each_loc)
    return total_emission


def viterbi(test_sample, initial_prob, transition_prob):
    '''

    :param test_sample: array of observed letters in ocr problem
    :param initial_prob: {state: probability}
    :param transition_prob:
    :return:

    test_sample: array of test sample splitted
    for eg: The sky is blue.
    ocr-> ['T','h','e','',....'e','.']
    pos-> ['The','sky',...'blue']

    initial_prob = {"The":0.5,
               "You":0.4
                  }

    transition_prob = {"The":{"sky":0.4}}
    '''

    costs = []
    for index, observed in enumerate(test_sample):
        if index == 0:
            '''
            Initial probability = emission*prior
            '''
            arr = {}
            for q in initial_prob:
                initial = -math.log(initial_prob[q])
                arr[q] = [emission_probability(observed, train_letters[q])/2 + initial, q]
            costs.append(arr)

        else:
            costs.append({})
            for q in initial_prob:
                min_arg = {}
                for i in initial_prob:
                    try:
                        trans_prob = -math.log(transition_prob[i][q])
                    except:
                        trans_prob = -math.log(low_prob)

                    min_arg[i] = costs[index - 1][i][0] + trans_prob

                key = min(min_arg, key=lambda x: min_arg[x])  # Find the key with min values in the dictionary

                costs[index][q] = [emission_probability(observed, train_letters[q]) + min_arg[key], key]

    '''
    Backtracking the best sequence
    '''
    sequence = []
    i = len(costs) - 1
    key = min(costs[i], key=lambda x: costs[i][x][0])
    while i >= 0:
        sequence.insert(0, key)
        key = costs[i][key][1]
        i -= 1

    return sequence


# Pre-computing priors and transition probabilities
initial_probability, transition_probability = train(read_data(train_txt_fname))

# Running the simple model
simple_model_result = simple_model(test_letters,train_letters,initial_probability)
simple_sentence = [heapq.heappop(letter_heap)[1] for letter_heap in simple_model_result]
print("Simple: " + "".join(simple_sentence))

# Running Viterbi on HMMs
sequence = viterbi(test_letters, initial_probability, transition_probability)
print("Viterbi: " + "".join(sequence))


# Print whatever gives the best answer
print("Final answer:")
print("".join(sequence))