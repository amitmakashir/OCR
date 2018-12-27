#!/usr/bin/python3
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# python ocr.py courier-train.png abc test-0-0.png
# Authors: (Amit Makashir, Rohit Bapat, Akshay Rathi)
# (based on skeleton code by D. Crandall, Oct 2018)
#

from PIL import Image, ImageDraw, ImageFont
import sys
import numpy as np
import heapq
import math

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25
Total = CHARACTER_WIDTH*CHARACTER_HEIGHT

def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    # print(im.size)
    # print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    # print(TRAIN_LETTERS)
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

def emission_probability(unmatched):
    m = 0.01 # Tune this as per your needs
    matched = Total - unmatched
    prob = ((1-m)**matched)*(m**unmatched)
    if prob == 0:
        return float("inf")
    else:
        return -math.log(prob)


#####
# main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)
# print(train_letters["A"])
## Below is just some sample code to show you how the functions above work. 
# You can delete them and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
# print("\n".join([ r for r in train_letters['U'] ]))

# Same with test letters. Here's what the third letter of the test data
#  looks like:
# print("\n".join([ r for r in test_letters[1] ]))
# print(train_letters)


# Simplified model
def simple_model(test_letters,train_letters):
    total_emission = []
    for test_letter in test_letters:
        emission_for_each_loc = []
        for letter, letter_array in train_letters.items():
            no_of_mismatched = sum(c1 != c2 for i, j in zip(letter_array, test_letter) for c1, c2 in zip(i, j))
            cost = emission_probability(no_of_mismatched)
            heapq.heappush(emission_for_each_loc,(cost,letter))
        total_emission.append(emission_for_each_loc)
    return total_emission


simple_model_result = simple_model(test_letters,train_letters)
simple_sentence = [heapq.heappop(letter_heap)[1] for letter_heap in simple_model_result]
print("".join(simple_sentence))