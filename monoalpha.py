#!/usr/bin/env python3

from ntpath import join
import random
import re
import csv
from math import ceil, sqrt
import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import time

#import cProfile

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


# === MONOALPHA CIPHER ===
# ========================

# Takes a List and returns a permutatation of it
def permutate(list):
    permutation = []
    for item in list:
        i = random.randint(0, len(permutation))
        permutation.insert(i, item)
    return permutation

# Takes a string and returns a permutatation of it
# (used for key generation)
def permutate_str(str):
    permutation = ''
    for letter in str:
        i = random.randint(0, len(str)-1)
        permutation = permutation + str[i]
        str = str[:i] + str[i+1:]
    return permutation

def encrypt(key, text, block_size = 5):
    cipher = ''
    text = re.sub('[^a-zA-Z]+', '', text) # filter only letters
    text = text.upper()
    idx = 1
    for char in text:
        char_i = ALPHABET.find(char)
        cipher = cipher + key[char_i]
        if idx % block_size == 0:
            cipher = cipher + ' '
        idx = idx + 1
    return cipher

def decrypt(key, cipher):
    text = ''
    cipher = re.sub('[^a-zA-Z]+', '', cipher) # filter only letters
    cipher = cipher.upper()
    for char in cipher:
        char_i = key.find(char)
        if char_i == -1:
            text = text + char.lower()
        else:
            text = text + ALPHABET[char_i]
    return text



# === FREQUENCY ANALYICS ===
# ==========================

# find N-grams in a string
def get_Ngrams(str, N):
    n_grams = []
    str = re.sub('[^a-zA-Z]+', '', str) # filter only letters
    for i in range(len(str) - N + 1):
        n_grams.append(str[i:i+N])
    return n_grams

# calculate how often a element in a list occurs
# returns a list of the form
# [[element1 , number of occurences], [element2, no_of_], ...]
def get_multiplicities(list):
    d = Counter(list)
    res = []
    for key in d:
        res.append([key, d[key]])
    return sorted(res, key=lambda x: -x[1] if len(x) > 0 else 0)

def load_freq_data():
    print("loading data..")
    with open ("data/letter_freq_en.csv") as f:
        letters_data = list(csv.reader(f, delimiter = ";"))
        for i in range(len(letters_data)):
            letters_data[i][1] = float(letters_data[i][1]) / 100
        letters_data = sorted(letters_data, key=lambda x: -x[1] if len(x) > 0 else 0)

    with open ("data/bigram_freq_en.csv") as f:
        bigram_data = list(csv.reader(f, delimiter = ";"))
        bigram_data = bigram_data[1:]
        for i in range(len(bigram_data)):
            bigram_data[i][1] = float(bigram_data[i][1]) / 100
        bigram_data = sorted(bigram_data, key=lambda x: -x[1] if len(x) > 0 else 0)

    with open ("data/trigram_freq_en.csv") as f:
        trigram_data = list(csv.reader(f, delimiter = ";"))
        trigram_data = trigram_data[1:]
        for i in range(len(trigram_data)):
            trigram_data[i][1] = float(trigram_data[i][1][0:-1]) / 100
            trigram_data[i][0] = trigram_data[i][0].lower()
        trigram_data = sorted(trigram_data, key=lambda x: -x[1] if len(x) > 0 else 0)

    with open ("data/quadgram_freq_en.csv") as f:
        quadgram_data = list(csv.reader(f, delimiter = ";"))
        quadgram_data = quadgram_data[1:]
        for i in range(len(quadgram_data)):
            quadgram_data[i][1] = float(quadgram_data[i][1][0:-1]) / 100
            quadgram_data[i][0] = quadgram_data[i][0].lower()
        quadgram_data = sorted(quadgram_data, key=lambda x: -x[1] if len(x) > 0 else 0)

    # with open("data/words_370k.txt") as f:
    #     words = f.readlines()
    # words = [_[:-1] for _ in words]

    # with open("data/word_freq_en.csv") as f:
    #     word_freq= list(csv.reader(f, delimiter = ";"))[1:]
    #     words = []
    #     for _ in word_freq:
    #         if (len(_[1]) != 1 or _[1] in ['i','a','o']):
    #             words.append(_[1])
    #     words = list(dict.fromkeys(words))
    #     words.append('is')
    #     words.append('an')
    #     words.append('done')
    #     words.append('was')
    #     words.append('s') #for plurals
    #     words.append('ed')
    #     words.append('planned')
    #     words.append('secret')
    #     words.append('para')
    #     words.append('longer')

    with open("data/words_freq.txt") as f:
        words = f.readlines()
    words = [_[:-1].replace("'","") for _ in words]

    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    # remove words with special characters and apply lower case
    words = [word.lower() for word in words if all((l.upper() in alphabet) for l in word)]
    # remnove single letter words
    words = [word for word in words if word not in ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n',\
                        'p', 'q', 'r', 't', 'v', 'w', 'x', 'z']]
    # remove less frequent two letter words
    words = [word for idx, word in enumerate(words) if (len(word) < 3 and idx < 2000) or len(word) >= 3 ]

    word_freq = {}
    for i, word in enumerate(words):
        # to ensure first occurance of word defines the frequency
        if word not in word_freq:
            word_freq[word] = i

    # for word in words:
    #         if any((l.upper() not in alphabet) for l in word):
    #             print(word)

    letter_dicts = []
    for word in words:
        for idx, char in enumerate(word):
            if len(letter_dicts) - 1 < idx:
                letter_dicts.append({})
            if char in letter_dicts[idx]:
                letter_dicts[idx][char].add(word)
            else:
                letter_dicts[idx][char] = {word}

    #     # sort words via first letter for faster search
    #     words_dict = {}
    #     for word in words:
    #         if word[0] in words_dict:
    #             words_dict[word[0]].append(word)
    #         else:
    #             words_dict[word[0]] = [word]

    # load forbidden quads
    with open("data/forbidden_quads.txt", "r") as f:
        forbidden_quads = set(f.readlines())
    forbidden_quads = [_[:-1] for _ in forbidden_quads]

    return letters_data, bigram_data, trigram_data, quadgram_data, letter_dicts, set(words), word_freq, forbidden_quads

# Takes a text as input and returns the multiplicities of the occuring 1-, 2-, 3-, and 4-grams.
def freq_ana(cipher):
    c_letters  = get_multiplicities(get_Ngrams(cipher, 1))
    c_bigrams  = get_multiplicities(get_Ngrams(cipher, 2))
    c_trigrams = get_multiplicities(get_Ngrams(cipher, 3))
    c_quadgrams = get_multiplicities(get_Ngrams(cipher, 4))
    return c_letters, c_bigrams, c_trigrams, c_quadgrams

def calc_sigma(key):
    if key not in ['1', '2', '3', '4']:
        return None
    sigma = 0
    tot_cngrams = CIPHER_LEN - int(key) + 1
    if len(DATA[key]) < len(C_DATA[key]):
        i_max = len(DATA[key])
    else:
        i_max = len(C_DATA[key])
    for i in range(i_max):
        c_freq = C_DATA[key][i][1] / tot_cngrams
        sigma += (c_freq - DATA[key][i][1])**2
    return sqrt(sigma)



# === CRACKING ===
# ================

class Guess:
    def __init__(self, dict = {}, redundancy = {}, route = [], route_idx = [], route_words = []):
        # contains the mapping of the guess. The keys get mapped to the values in encryption.
        self.dict = dict.copy()
        self.map_fitness = 0
        self.redundancy = redundancy.copy()
        self.avg_redundancy = -1
        # contains pairs of ngrams. [cipher_ngram, plaintxt_ngram]. The cipher n_gram is guessed to equal the plaintxt_ngram.
        self.route = route.copy()
        # contains the indices of the ngrams in route. 
        self.route_idx = route_idx.copy()
        # a value that estimates the qualitiy of the ngram guesses
        self.ngram_fitness = 0
        # contains a list of pairs [word, letters_mapped, word_frequency], where letters_mapped is the number of letters that are added to the guess
        # for the word.
        self.route_words = route_words.copy()
        # concatinates all words in route_words. used to identify equivalent guesses.
        self.route_txt = ""
        # a value to estimate the quality of the word guesses. The ratio gaps to word length.
        self.words_fitness = 0
        # the number of words in route that resulted in a guessed map (updated along with words_fitness)
        self.words_length = 0
        # the avg of the word frequency in route_words
        self.avg_word_freq = 0


    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
        #return (self.dict == other.dict) and (self.route == other.route) and (self.route_txt == other.route_txt)
    
    def __hash__(self) -> int:
        return hash(str(self.dict) + str(self.route) + self.route_txt)

    def jsonify(self):
        return {
            "key": self.gen_key(),
            "route": self.route,
            "route_idx": self.route_idx,
            "ngram_fitness": self.ngram_fitness
        }
    def update_all(self):
        self.update_avg_redundancy()
        self.update_ngram_fitness()
        self.update_route_txt()
        self.update_words_fitness()

    def add(self, char1, char2):
        if char1 in self.dict:
            if self.dict[char1] == char2:
                self.redundancy[char1] += 1
                return 1
            else:
                return 0
        elif char2 in self.dict.values():
            return 0
        else:
            self.dict[char1] = char2
            self.redundancy[char1] = 1
            return 1

    def gen_key(self):
        alphabet_lst = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        key = ['-' for i in range(len(alphabet_lst))]
        for char in self.dict:
            try:
                idx = alphabet_lst.index(char)
            except ValueError:
                print(char, self.dict)
                return None
            key[idx] = self.dict[char]
        return ''.join(key)

    def consistant_key(self, key: str):
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i in range(len(key)):
            target = key[i]
            source = alphabet[i]
            try:
                if (self.dict[source] != target and target != '-'):
                    return False
            except:
                continue
        return True

    def consistant(self, other):
        domain_self = set(self.dict.keys())
        domain_other = set(other.dict.keys())
        domain_overlap = domain_self.intersection(domain_other)
        domain_overlap_size = len(domain_overlap)
        image_self = set(self.dict.values())
        image_other = set(other.dict.values())
        image_overlap = image_self.intersection(image_other)
        image_overlap_size = len(image_overlap)

        if domain_overlap_size == 0 and image_overlap_size == 0:
            return True
        elif (domain_overlap_size == 0 and image_overlap_size != 0) or (domain_overlap_size != 0 and image_overlap_size ==0):
            return False
        else:
            for src in domain_overlap:
                if self.dict[src] != other.dict[src]:
                    return False
        return True

    def update_map_fitness(self, maps):
        if len(self.dict) != 0:
            self.map_fitness = 0;
            for m in maps:
                if m[0] in self.dict:
                    if m[1] == self.dict[m[0]]:
                        self.map_fitness += m[2] * self.redundancy[m[0]]
            self.map_fitness /= len(self.dict)

    def update_avg_redundancy(self):
        if len(self.dict) != 0:
            self.avg_redundancy = sum(self.redundancy.values()) / len(self.dict)

    def update_ngram_fitness(self):
        sum = 0
        i = 0
        while i<len(self.route_idx) and len(self.route_idx[i])==2:
            idx = self.route_idx[i]
            sum += (idx[0] - idx[1])**2 / (i + 1)**2
            i += 1
        if len(self.route_idx) != 0:
            sum /= len(self.route_idx)**2
        self.ngram_fitness = sum

    def update_route_txt(self):
        self.route_txt = ""
        for node in self.route_words:
            self.route_txt += node[0]
    
    def update_words_fitness(self):
        if len(self.route_words) == 0:
            self.words_fitness = 0
            return 0
        res = 0
        length = 0
        for node in self.route_words:
            if node[1] != 0:
                res += node[1] / len(node[0])**2
                length += 1
        if length == 0:
            self.words_fitness = 0
            return 1
        self.words_fitness = res / length
        self.words_length = length
        self.avg_word_freq = sum([n[2] for n in self.route_words]) / len(self.route_words)
        return 1


def gen_consistant_guesses(start_guess: Guess, n: str, depth, nodes):
    guesses = []
    try:
        c_ngram = C_DATA[n][depth][0]
    except Exception as e:
        print(e)
        print(f"{n}-grams not supported")
        return -1

    for node in nodes:
        next_guess = Guess(start_guess.dict, start_guess.redundancy, start_guess.route, start_guess.route_idx)
        consistant = True
        # add ngram to guess
        ngram = DATA[n][node][0]
        for i in range(len(ngram)):
            char1 = ngram[i].upper()
            char2 = c_ngram[i]
            if not next_guess.add(char1, char2):
                consistant = False
                break
        if consistant:
            next_guess.route.append([c_ngram, ngram])
            next_guess.route_idx.append([depth, node])
            next_guess.update_ngram_fitness()
            guesses.append(next_guess)
    return guesses

# recursively generates key-guesses starting with keyguess. 
# nodes is a list of indices of the list of N-grams in the english language
# MAX-DEPTH gives termination condition
# FILTER_BRANCH: controls the number of childs for each node. 
# DEPTH: controls the start DEPTH 
def ngram_search(n: str, keyguess: Guess, nodes, MAX_DEPTH = 3, FILTER_BRANCH = 1, DEPTH = 0):
    if DEPTH >= MAX_DEPTH:
        GUESSES.append(keyguess)
        return 1
    else:
        lvl_guesses = []
        while len(lvl_guesses)==0 and DEPTH < len(C_DATA[n]):
            lvl_guesses= gen_consistant_guesses(keyguess, n, DEPTH, nodes)
            DEPTH += 1
        lvl_guesses.sort(key=lambda x: x.ngram_fitness)
        for i in range(ceil(len(lvl_guesses) * FILTER_BRANCH)):
            guess = lvl_guesses[i]
            next_nodes = nodes.copy()
            next_nodes.remove(nodes[i])
            ngram_search(n, guess, next_nodes, MAX_DEPTH, FILTER_BRANCH, DEPTH)
        # include the start guess
        if len(keyguess.route) != 0:
            #FILTER_BRANCH *= 1/DEPTH
            ngram_search(n, keyguess, nodes, MAX_DEPTH, FILTER_BRANCH, DEPTH)

# finds the longest word in a cloze
def longest_word(cloze: str):
    candidates = get_candidates_cloze(cloze)
    max_length = max([len(w) for w in candidates])
    (pos, longest) = (len(cloze), '')
    for word in candidates:
        word = word.upper()
        if len(word) == max_length:
            pos_tmp = cloze.find(word)
            if pos_tmp < pos:
                longest = word
                pos = pos_tmp
    return (pos, longest)

# finds the next complete word in a clozw
def next_word(cloze: str):
    length = len(cloze)
    # find words in cloze
    candidates = list(get_candidates_cloze(cloze))
    if not candidates:
        print("next_word: WARNING no canditates in cloze")
        return(0, '')
    frequencies = []
    for word in candidates:
        frequencies.append(WORD_FREQ[word])
    candidates = [x for _, x in sorted(zip(frequencies, candidates))]
    # get the position of the found words
    # (pos, next) = (length, '')
    freq = max(frequencies)
    for word in candidates:
        word = word.upper()
        pos_tmp = cloze.find(word)
        if pos_tmp<length*0.3 and len(word)>3:
            return (pos_tmp, word)
    # print("WARNING: next_word did not find a word with at least 4 letters")
    return (cloze.find(candidates[0]), candidates[0])


# splits the text in blocks of length rad and returns the position of the block, 
# where the most characters coincide with target.
def find_dense_area(text: str, target: str, rad: int):
    N = 0
    N_max = 0
    pos_max = []
    for i in range(len(text) - rad):
        area = text[i:i+rad]
        for char in area:
            if char == target and prev_char != target:
                N += 1
            prev_char = char
        if N > N_max:
            N_max = N
            pos_max = [i]
        elif N==N_max and N != 0:
            pos_max.append(i)
        N = 0
    return pos_max

def get_candidates_cloze(cloze:str):
    length = len(cloze)
    # find words in cloze
    cloze_blocks = cloze.split('-')
    cloze_blocks = [b for b in cloze_blocks if len(b)>2]
    candidates = set()
    for b in cloze_blocks:
        for i in range(len(b)-2):
            block_candidates = get_candidates(b[i:])
            block_candidates = [c for c in block_candidates if len(c) <= len(b[i:]) and len(c)>2]
            candidates = candidates.union(set(block_candidates))
    return candidates

def get_candidates(pattern: str):
    pattern = pattern.lower()
    sets = []
    for idx, char in enumerate(pattern):
        if char != '-' and idx < len(WORDS):
            try:
                sets.append(WORDS[idx][char])
            except (KeyError, IndexError) as e:
                # print("WARNING: " + str(e))
                break
        else:
            sets.append(WORDS_SET)
    candidates = set()
    for idx, x in enumerate(sets):
        if idx + 1 == len(sets):
            partial_matches = sets[0].intersection(*sets)
        else:
            partial_matches = set([_ for _ in sets[0].intersection(*sets[0:idx + 1]) if len(_) == idx + 1])
        candidates = candidates.union(partial_matches)
    return candidates

# starts a recursive search for words in cloze. Genereates keyguesses by filling in the gaps in the cloze.
def find_words(cloze: str, keyguess: Guess, cipher: str, DEPTH = 0, PRINT = False, old_guesses = []):
    if DEPTH == 0 and PRINT:
        print("find_words:")
        print("  ", keyguess.gen_key())
        print("  ", cipher[:50])
        print("  ", cloze[:50])
    if DEPTH > PARAMETER['DEPTH_W'] or \
       len(keyguess.dict) == 26 or \
       len(cloze) < 20 or \
       cloze.find('-') == -1:
        guesses.append(keyguess)
        # print('\r', len(guesses), DEPTH, len(keyguess.dict), f"{len(cloze)/CIPHER_LEN: .2%}", end="")
        return True
    if len(guesses) > 1000:
        if PRINT:
            print("Found more than 1000 new guesses, canceling recursion")
        return True

    matched_words = []
    new_guesses = []

    candidates = list(get_candidates(cloze[:20]))
    # candidates = list(WORDS_SET)

    for word in candidates:
        match = True
        maps = {}
        for i in range(len(word)):
            char = word[i]
            if cloze[i] == '-':
                if char.upper() not in maps.keys() and cipher[i].upper() not in maps.values():
                    maps[char.upper()] = cipher[i].upper()
                elif char.upper() in maps.keys():
                    match = (maps[char.upper()] == cipher[i].upper())
                    if not match: 
                        break
                elif char.upper() not in maps.keys():
                    match = False
                    break
            else:
                if char.upper() != cloze[i]:
                    match = False
                    break
        # longer words with fewer '-' that still match the cloze is also a
        # measure for confidence
        #
        # save all matched words for a level and check
        # 1) if a smaller word is part of a longer matched word -> prefer the longer
        #    as long as, if the shorter one is removed from the longer, the remaining letters
        #    are not a word nor the beginning of a word
        # 2) prefer long words with few gaps guessed
        if match:
            consistant = True
            new_keyguess = Guess(keyguess.dict, keyguess.redundancy, keyguess.route, keyguess.route_idx, keyguess.route_words)
            for key in maps:
                if not new_keyguess.add(key, maps[key]):
                    consistant = False # due to inconsitency with keyguess
                    break

            # check for forbidden quads
            if consistant and not forb_quads_in_cloze(new_keyguess, cipher):
                matched_words.append(word)
                new_keyguess.route_words.append([word, len(maps), WORD_FREQ[word]])
                new_keyguess.update_route_txt()
                new_keyguess.update_words_fitness()
                new_guesses.append(new_keyguess)

    new_guesses_sort = sorted(new_guesses, key=lambda x: x.words_fitness * x.avg_word_freq)

    # start recursion on selected new_keyguesses
    for g in new_guesses_sort:
        if PRINT:
            print(DEPTH, g.route_words, g.words_fitness)
        same_txt = False
        for guess in old_guesses:
            if guess.route_txt == g.route_txt:
                same_txt = True
                break
        if (g.words_fitness < 0.6 or g.words_length < 2) and not same_txt:
            new_cloze = re.sub('[a-z]','-',decrypt(g.gen_key(), cipher))
            find_words(new_cloze[len(g.route_words[-1][0]):], g, cipher[len(g.route_words[-1][0]):], DEPTH + 1, PRINT, new_guesses)

def forb_quads_in_cloze(g: Guess, cipher: str):
    contains = False
    cloze = re.sub('[a-z]','-',decrypt(g.gen_key(), cipher))
    cloze_blks = [b.lower() for b in cloze.split('-') if len(b)>3]
    cloze_quads = set()
    for blk in cloze_blks:
        for i in range(4, len(blk)):
            cloze_quads.add(blk[i-4:i])
    return cloze_quads.intersection(FORB_QUADS)


def gap_possibilites(cloze, cipher, guesses):
    pos = 0
    possibilities = {}
    while pos < len(cloze):
        pos = cloze.find('-',pos)
        if pos == -1: break
        end = pos + 1
        gap = '-'
        while True:
            if end < len(cloze):
                if cloze[end] == '-':
                    end += 1
                    gap += '-'
                else: break
            else: break
        for g in guesses:
            cloze_g = re.sub('[a-z]','-',decrypt(g.gen_key(), cipher))
            gap_guess = cloze_g[pos:end]
            if gap_guess.find('-') == -1:
                if pos not in possibilities.keys():
                    possibilities[pos] = [[gap_guess, 1 / len(guesses)]]
                else:
                    found = False
                    for item in possibilities[pos]:
                        if item[0] == gap_guess:
                            found = True
                            break
                    if found:
                        item[1] += 1/len(guesses)
                    else:
                        possibilities[pos].append([gap_guess, 1/len(guesses)])
        pos += len(gap)
    return possibilities


class Crack:
    def __init__(self, guesses = [Guess()]):
        self.guesses = guesses
        self.min = Guess()
        self.maps = []
        self.bad_maps = []
        self.consistancies = [None for g in guesses]

    def add(self, g: Guess):
        self.guesses.append(g)

    def add_unique(self, g: Guess):
        for guess in self.guesses:
            if guess.gen_key() == g.guess.gen_key():
                return False
        self.guesses.append(g)
        return True

    def shrink(self):
        print("shrinking..")
        guesses = []
        keys = [g.gen_key() for g in self.guesses]
        uniq_keys = []
        for i in range(len(self.guesses)):
            if keys[i] not in uniq_keys:
                g = self.guesses[i]
                red = list(g.redundancy.values())
                for j in range(i+1, len(self.guesses)):
                    if keys[i] == keys[j]:
                        red = [x+y for x,y in zip(red, self.guesses[j].redundancy.values())]
                red_dict = g.redundancy
                k = 0
                for key in g.redundancy:
                    red_dict[key] = red[k]
                    k += 1
                guess = Guess(g.dict, red_dict, g.route, g.route_idx, g.route_words)
                guess.update_all()
                guesses.append(guess)
                uniq_keys.append(keys[i])
        self.guesses = guesses

    def get_min(self):
        min = Guess(self.guesses[0].dict)
        for g in self.guesses:
            if not g.consistant(min):
                keys = list(min.dict.keys()).copy()
                for src in keys:
                    if src not in g.dict.keys():
                        del min.dict[src]
                    elif g.dict[src] != min.dict[src]:
                        del min.dict[src]
            elif len(min.dict) > len(g.dict):
                min = g
        self.min = min
        return self.min

    def minimize(self):
        print("minimizing..")
        self.get_min()
        if self.min.dict == {}:
            return False
        else:
            self.guesses = [self.min]
            return True

    def get_maps_w_redundancy(self):
        res = []
        for g in self.guesses:
            for src in g.dict:
                res.append((src, g.dict[src]))
        # maps = [[src, tgt, red], ...]
        count = Counter(res)
        maps = []
        for map in count:
            maps.append([map[0], map[1], count[map]])
        maps = sorted(maps, key=lambda x: -x[2])
        self.maps = maps
        return self.maps

    def calc_map_fitness(self):
        for g in self.guesses:
            g.update_map_fitness(self.maps)

    def filter(self, ngram_fitness_filter, length_filter, map_filter):
        print("filtering ngrams..")
        l = len(self.guesses)
        fitness = sorted(self.guesses, key=lambda x: x.ngram_fitness)
        fitness_threshold = ceil(ngram_fitness_filter * l)
        fitness = fitness[ : fitness_threshold ]
        length = sorted(self.guesses, key=lambda x: -len(x.dict))
        length_threshold = ceil(length_filter * l)
        length = length[ : length_threshold ]
        map_fitness = sorted(self.guesses, key=lambda x: -x.map_fitness)
        map_fitness_threshold = ceil(map_filter * l)
        map_fitness = map_fitness[ : map_fitness_threshold ]
        filtered = list(set(fitness) & set(length) & set(map_fitness))
        self.guesses = sorted(filtered, key=lambda x: x.ngram_fitness)

    def word_filter(self, length_filter, words_fitness_filter):
        print("filtering words..")
        l = len(self.guesses)
        length = sorted(self.guesses, key=lambda x: -len(x.dict))
        length_threshold = ceil(length_filter * l)
        length = length[ : length_threshold ]
        fitness = sorted(self.guesses, key=lambda x: x.words_fitness)
        fitness_threshold = ceil(words_fitness_filter * l)
        fitness = fitness[ : fitness_threshold ]
        filtered = list(set(fitness) & set(length))
        self.guesses = sorted(filtered, key=lambda x: x.words_fitness)

    # use self.maps to shave off mappings for each guess that are not often globally.
    def shave(self, filter):
        print("shaving..")
        maps_filtered = self.maps[:ceil(len(self.maps) * filter)]
        shaved_guesses = []
        for g in self.guesses:
            guess = Guess(g.dict, g.redundancy)
            for src in g.dict:
                tgt = g.dict[src]
                found = False
                for m in maps_filtered:
                    if src==m[0] and tgt==m[1]:
                        found = True
                        break
                if not found:
                    del guess.dict[src]
                    del guess.redundancy[src]
            guess.update_all()
            shaved_guesses.append(guess)
        return shaved_guesses

    def calc_consistancies(self):
        self.consistancies = [0 for g in self.guesses]
        keys = [g.gen_key() for g in self.guesses]
        for i in range(len(self.guesses)):
            for j in range(i+1, len(self.guesses)):
                print(f"\r calc_consistancies: {i}", end="")
                if self.guesses[i].consistant_key(keys[j]):
                    self.consistancies[i] += 1
                    self.consistancies[j] += 1
        print()

    def get_adj_matrix(self):
        self.calc_consistancies()
        L = len(self.guesses)
        adj = np.zeros((L,L))
        no_consistancies = []
        for i in range(L):
            if self.consistancies[i] == 0:
                no_consistancies.append(i)
            else:
                for j in range(i+1, L):
                    if self.guesses[i].consistant(self.guesses[j]):
                        adj[i][j] = 1
                        adj[j][i] = 1
        return np.delete(np.delete(adj, no_consistancies, axis=0), no_consistancies, axis=1)
        return adj

    def get_consistant_guesses(self):
        res = []
        checked = []
        for i in range(len(self.guesses)):
            if i not in checked:
                checked.append(i)
                consistant_guesses = [self.guesses[i]]
                for j in range(i+1, len(self.guesses)):
                    if self.guesses[i].consistant(self.guesses[j]):
                        consistant_guesses.append(self.guesses[j])
                        checked.append(j)
                res.append(consistant_guesses)
        return res

    def gen_guesses_ngram(self, n, log: bool):
        n = str(n)
        try:
            nodes = list(range(len(DATA[n])))
            depth = PARAMETER['DEPTH_'+n]
            filter_branch = PARAMETER['FILTER_BRANCH_'+n]
            filter_total = PARAMETER['FILTER_'+n]
        except Exception as e:
            print(e)
            print(f"{n}-grams not supported")
            return -1

        global GUESSES
        GUESSES = self.guesses.copy()
        if log:
            print(f"=== {n}-GRAMS ===")
            sigma = calc_sigma(n)
            print(f"SIGMA: {sigma: .4f}")
        for i in range(len(self.guesses)):
            ngram_search(n, self.guesses[i], nodes, depth, filter_branch)
            if log:
                print(f"\r progress: {i+1}/{len(self.guesses)}    new guesses: {len(GUESSES)}", end="")

        self.guesses = GUESSES.copy()
        self.get_maps_w_redundancy()
        self.calc_map_fitness()
        self.consistancies = [None for g in self.guesses]

        if log:
            print()
            self.print_guesses()

    def get_guesses_words(self, cipher, mode="start"):
        start = time.perf_counter()
        guesses_w_words_found = []
        log_flag = False
        for i, guess in enumerate(self.guesses):
            if mode == "start":
                if len(guesses_w_words_found) > 0:
                    # print(f"canceling search for new guesses via words, found {len(guesses_w_words_found)}")
                    print("found guesses, canceling search.")
                    break
                cur_cloze = re.sub('[a-z]','-',decrypt(guess.gen_key(), cipher))
                cur_cipher = cipher
                max_tries = 6
                PARAMETER['DEPTH_W'] = 15
            elif mode == "next_longest":
                if len(guesses_w_words_found) > 200:
                    print(f"canceling search for new guesses via words, found {len(guesses_w_words_found)}")
                    break
                route_char_len = 0
                # for node in guess.route_words:
                #     route_char_len += len(node[0])
                cur_cipher = cipher[route_char_len:]
                cur_cloze = re.sub('[a-z]','-',decrypt(guess.gen_key(), cur_cipher))
                (pos, word) = longest_word(cur_cloze)
                cur_cloze = cur_cloze[pos + len(word):]
                cur_cipher = cur_cipher[pos + len(word):]
                max_tries = 10
                PARAMETER['DEPTH_W'] = 6
                # print()
                # print(word)
                # print(cur_cloze[:100])
                # print(cur_cipher[:100])
            elif mode == "next_gap":
                if len(guesses_w_words_found) > 200:
                    print(f"canceling search for new guesses via words, found {len(guesses_w_words_found)}")
                    break
                cur_cloze = re.sub('[a-z]','-',decrypt(guess.gen_key(), cipher))
                first_gap_pos = cur_cloze.find('-')
                chunk_size = 30
                if first_gap_pos < chunk_size:
                    chunk_start = 0
                else:
                    chunk_start = first_gap_pos - chunk_size
                (pos, word) = longest_word(cur_cloze[chunk_start:first_gap_pos])
                if word == '':
                    print("WARNING: No longest word in chunk.")
                start_pos = first_gap_pos - chunk_size + pos + len(word)
                cur_cipher = cipher[start_pos:]
                cur_cloze = re.sub('[a-z]','-',decrypt(guess.gen_key(), cur_cipher))
                max_tries = 10
                PARAMETER['DEPTH_W'] = 6

            start_pos = []
            global guesses
            guesses = []
            avg = 0
            start_pos.append(0)
            print(f"\r new keys found: {len(guesses_w_words_found)}   keys checked: {i}/{len(self.guesses)}    current key: {guess.gen_key()}", end="")
            # add criteria to continue searching.
            # 1) bad word_fitness (ratio gaps filled to word length)
            # 2) word frequencies
            while (len(guesses) == 0 or avg>0.65):
                if len(start_pos) > max_tries: # max(start_pos) > (len(cipher)*0.95) :
                    break
                guesses = []
                # print(start_pos)
                find_words(cur_cloze, guess, cur_cipher, 0, log_flag)
                # find longest complete word in cloze
                (pos, next) = next_word(cur_cloze[1:])
                start_pos.append(start_pos[-1] + pos + 1 + len(next))
                # start new_cloze behind that word
                cur_cloze = cur_cloze[pos + 1 + len(next):]
                cur_cipher = cur_cipher[pos + 1 + len(next):]
                avg = 0
                if len(guesses) != 0:
                    print()
            guesses_w_words_found += guesses
        print()
        stop = time.perf_counter()
        guesses_w_words_found = sorted(guesses_w_words_found, key=lambda x: x.words_fitness)
        print(f"Words Search took {stop-start: .1f} seconds")
        return guesses_w_words_found

    def print_guesses(self, dev=True):
        c = 0
        print(f"                       KEY   ngft   red    mpft   c   csts   wdft   wfreq   len")
        for i in range(len(self.guesses)):
            g = self.guesses[i]
            g.update_avg_redundancy()
            if dev:
                global key
                if g.consistant_key(key) or i < 20:
                    print(f"{g.gen_key()}  {g.ngram_fitness: .2f}  {g.avg_redundancy: .2f}  {g.map_fitness: .2f}  {g.consistant_key(key): 1}  {self.consistancies[i]}  {g.words_fitness*100: .2f}  {g.avg_word_freq: .2f}  {len(g.dict)}")
                    # for src in ALPHABET:
                    #     if src in g.redundancy.keys():
                    #         print(g.redundancy[src], end="")
                    #     else:
                    #         print(0, end="")
                    # print()
                    if g.consistant_key(key): c += 1
            else:
                if i < 20:
                    print(f"{g.gen_key()}  {g.ngram_fitness: .2f}  {g.avg_redundancy: .2f}  {g.map_fitness: .2f}  {self.consistancies[i]}  {g.words_fitness*100: .2f}  {g.avg_word_freq: .2f}  {len(g.dict)}")

        if dev: print(f"{c} keys of {len(self.guesses)} consistant in self.guesses")
        print()

    # export guesses to json
    def save_guesses(self, outname, number):
        with open("out/" + outname +".json", 'w') as f:
            out = {}
            for i,g in enumerate(self.guesses):
                if i > number:
                    break
                out[str(i)] = g.jsonify()
            # for i in range(len(self.guesses)):
            #     out[str(i)] = self.guesses[-i-1].jsonify()
            json.dump(out, f)

    def get_bad_maps_from_forb_quad(self):
        tmp_bad_maps = []
        for g in self.guesses:
            forb_quads = forb_quads_in_cloze(g, cipher)
            cloze = re.sub('[a-z]','-',decrypt(g.gen_key(), cipher))
            for forb_quad in forb_quads:
                pos = cloze.find(forb_quad.upper())
                for i in range(pos, pos+4):
                    tmp_bad_maps.append([cloze[i], cipher[i]])
        # clean dublicates and count occurances in tmp_bad_maps
        bad_maps = []
        counts = []
        for m in tmp_bad_maps:
            if m not in bad_maps:
                bad_maps.append(m)
                counts.append(1)
            else:
                counts[bad_maps.index(m)] += 1

        for i,m in enumerate(bad_maps):
            m.append(counts[i])

        bad_maps = sorted(bad_maps, key = lambda x: -x[2])

        self.bad_maps = bad_maps
        return self.bad_maps

    def delete_guesses_forb_quads(self, cipher):
        print("deleting guesses that produce forbidden quads..")
        for g in self.guesses:
            if forb_quads_in_cloze(g, cipher):
                # print(g.gen_key(), "produces forbidden quads, removing..")
                self.guesses.remove(g)


def crack(cipher,  p = {
        "SEARCH_3": True,
        "DEPTH_3": 5,
        "FILTER_BRANCH_3": 1,
        "FILTER_3": 1,
        "SEARCH_2": True,
        "DEPTH_2": 9,
        "FILTER_BRANCH_2": 0.7,
        "FILTER_2": 1,
        "SEARCH_1": False,
        "DEPTH_1": 8,
        "FILTER_BRANCH_1": 0.5,
        "FILTER_1": 1,
        "DEPTH_W": 17},
          dev = True):

    # change working directory
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    cipher = re.sub(r'[^A-Z]', '', cipher.upper())
    print(cipher)

    global CIPHER_LEN
    CIPHER_LEN = len(cipher)

    global LETTERS, BIGRAMS, TRIGRAMS, QUADGRAMS, WORDS, WORDS_SET, WORD_FREQ, FORB_QUADS
    (LETTERS, BIGRAMS, TRIGRAMS, QUADGRAMS, WORDS, WORDS_SET, WORD_FREQ, FORB_QUADS) = load_freq_data()
    # print(len(FORB_QUADS))
    global DATA
    DATA = {}
    DATA['1'] = LETTERS
    DATA['2'] = BIGRAMS
    DATA['3'] = TRIGRAMS
    DATA['W'] = WORDS

    # Parameter
    global PARAMETER
    SEARCH_W = True
    DECRYPT_GUESS = True
    PARAMETER = p

    global C_DATA
    C_DATA = {}

    global avg_fitness

    if (dev):
        global key
        key = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        print(key)

        # DEBUGGING

        # key_dict = {}
        # key_dict['A'] = 'A'
        # key_dict['D'] = 'D'
        # key_dict['E'] = 'E'
        # key_dict['H'] = 'H'
        # key_dict['I'] = 'I'
        # key_dict['N'] = 'N'
        # key_dict['T'] = 'T'
        # key_dict['S'] = 'S'
        # g = Guess(key_dict)
        # cloze = re.sub('[a-z]','-',decrypt(g.gen_key(), cipher))
        # # print(cipher)
        # return None

    if(p['SEARCH_3']):
        C_DATA['3'] = get_multiplicities(get_Ngrams(cipher, 3))

        CRACK = Crack([Guess()])

        CRACK.gen_guesses_ngram(3, True)

        CRACK.filter(0.05,1,1)
        CRACK.get_maps_w_redundancy()
        CRACK.calc_map_fitness()
        CRACK.print_guesses()

        if dev:
            fitness = [g.ngram_fitness for g in CRACK.guesses]
            length = [len(g.route_idx) for g in CRACK.guesses]
            plt.scatter(fitness , length)
            fitness_consistant = [g.ngram_fitness for g in CRACK.guesses if g.consistant_key(key)]
            length_consistant = [len(g.route_idx) for g in CRACK.guesses if g.consistant_key(key)]
            plt.scatter(fitness_consistant , length_consistant)
            plt.savefig("3-gram_fitness_length.jpg")

            plt.figure()
            avg_redundancy = [g.avg_redundancy for g in CRACK.guesses]
            plt.scatter(fitness, avg_redundancy)
            avg_redundancy_consistant = [g.avg_redundancy for g in CRACK.guesses if g.consistant_key(key)]
            plt.scatter(fitness_consistant , avg_redundancy_consistant)
            plt.savefig("3-gram_fitness_avg_redundancy.jpg")

            plt.figure()
            map_fitness = [g.map_fitness for g in CRACK.guesses]
            plt.scatter(fitness, map_fitness)
            map_fitness_consistant = [g.map_fitness for g in CRACK.guesses if g.consistant_key(key)]
            plt.scatter(fitness_consistant , map_fitness_consistant)
            plt.savefig("3-gram_fitness_map_fitness.jpg")

        CRACK.filter(1,1,0.5)
        CRACK.get_maps_w_redundancy()
        CRACK.calc_map_fitness()

        if dev:
            plt.figure()
            fitness = [g.ngram_fitness for g in CRACK.guesses]
            map_fitness = [g.map_fitness for g in CRACK.guesses]
            plt.scatter(fitness, map_fitness)
            map_fitness_consistant = [g.map_fitness for g in CRACK.guesses if g.consistant_key(key)]
            fitness_consistant = [g.ngram_fitness for g in CRACK.guesses if g.consistant_key(key)]
            plt.scatter(fitness_consistant , map_fitness_consistant)
            plt.savefig("3-gram_fitness_map_fitness_2.jpg")

        CRACK.shrink()
        CRACK.filter(1,1,0.5)
        CRACK.get_maps_w_redundancy()
        CRACK.calc_map_fitness()

        if dev:
            plt.figure()
            fitness = [g.ngram_fitness for g in CRACK.guesses]
            map_fitness = [g.map_fitness for g in CRACK.guesses]
            plt.scatter(fitness, map_fitness)
            fitness_consistant = [g.ngram_fitness for g in CRACK.guesses if g.consistant_key(key)]
            map_fitness_consistant = [g.map_fitness for g in CRACK.guesses if g.consistant_key(key)]
            plt.scatter(fitness_consistant , map_fitness_consistant)
            plt.savefig("3-gram_fitness_map_fitness_3.jpg")

            plt.figure()
            avg_redundancy = [g.avg_redundancy for g in CRACK.guesses]
            plt.scatter(fitness, avg_redundancy)
            avg_redundancy_consistant = [g.avg_redundancy for g in CRACK.guesses if g.consistant_key(key)]
            plt.scatter(fitness_consistant , avg_redundancy_consistant)
            plt.savefig("3-gram_fitness_avg_redundancy.jpg")
            CRACK.print_guesses()

            maps_filtered = [CRACK.maps[i] for i in range(ceil(len(CRACK.maps) * 0.6)) ]
            images = {}
            domains = {}
            for m1 in maps_filtered:
                if m1[0] not in images:
                    images[m1[0]] = []
                    tot = 0
                    for m2 in maps_filtered:
                        if m1[0] == m2[0]:
                            images[m1[0]].append([m2[1], m2[2]])
                            tot += m2[2]
                    for img in images[m1[0]]:
                        img.append(img[-1] / tot)
                if m1[1] not in domains:
                    domains[m1[1]] = []
                    tot = 0
                    for m2 in maps_filtered:
                        if m1[1] == m2[1]:
                            domains[m1[1]].append([m2[0], m2[2]])
                            tot += m2[2]
                    for dom in domains[m1[1]]:
                        dom.append(dom[-1] / tot)

            for m in maps_filtered:
                for img_data in images[m[0]]:
                    if img_data[0] == m[1]:
                        weight = img_data[2]
                        break
                for dom_data in domains[m[1]]:
                    if dom_data[0] == m[0]:
                        weight *= dom_data[2]
                        break
                m.append(weight)

            maps_filtered = sorted(maps_filtered, key=lambda x: -x[3]*x[2])
            for m in maps_filtered:
                print(f"{m[0]} {m[1]} {m[2]: 4} {m[3]: .6f}")
            print()


            printed= []
            for m1 in maps_filtered:
                frequencies = []
                for m2 in maps_filtered:
                    if m1[0] == m2[0] and m2[0] not in printed:
                        print(f"{m2[0]} {m2[1]} {m2[2]: 4}", end=" ")
                        for i in range(round(m2[2]/10)):
                            print("X", end="")
                        print()
                        frequencies.append(m2[2])
                diffs = []
                for i in range(len(frequencies) - 1):
                    diffs.append(frequencies[0] - frequencies[i+1])
                if m1[0] not in printed:
                    printed.append(m1[0])
                    # frequencies = list(map(lambda x: x/max(frequencies), frequencies))
                    print(f"mean: {np.mean(frequencies) : .2f}    std: {np.std(frequencies) : .2f}   var: {np.var(frequencies) : .2f}")
                    print(f"DIFFS mean: {np.mean(diffs) : .2f}    std: {np.std(diffs) : .2f}   var: {np.var(diffs) : .2f}")
                    print()

            printed= []
            for m1 in maps_filtered:
                frequencies = []
                for m2 in maps_filtered:
                    if m1[1] == m2[1] and m2[1] not in printed:
                        print(f"{m2[0]} {m2[1]} {m2[2]: 4}", end=" ")
                        for i in range(round(m2[2]/10)):
                            print("X", end="")
                        print()
                        frequencies.append(m2[2])
                diffs = []
                for i in range(len(frequencies) - 1):
                    diffs.append(frequencies[0] - frequencies[i+1])
                if m1[1] not in printed:
                    printed.append(m1[1])
                    print(f"mean: {np.mean(frequencies) : .2f}    std: {np.std(frequencies) : .2f}   var: {np.var(frequencies) : .2f}")
                    print(f"DIFFS mean: {np.mean(diffs) : .2f}    std: {np.std(diffs) : .2f}   var: {np.var(diffs) : .2f}")
                    print()

        SHAVED = Crack(CRACK.shave(0.9))
        SHAVED.shrink()
        SHAVED.get_maps_w_redundancy()
        SHAVED.calc_map_fitness()

        if dev:
            plt.figure()
            length = [len(g.dict) for g in SHAVED.guesses]
            map_fitness = [g.map_fitness for g in SHAVED.guesses]
            plt.scatter(length, map_fitness)
            length_consistant = [len(g.dict) for g in SHAVED.guesses if g.consistant_key(key)]
            map_fitness_consistant = [g.map_fitness for g in SHAVED.guesses if g.consistant_key(key)]
            plt.scatter(length_consistant , map_fitness_consistant)
            plt.savefig("3-gram_length_map_fitness_shaved.jpg")

            plt.figure()
            size = list(map(lambda x: (x - min(length))/(max(length)- min(length)) * 100 + 10, length))
            avg_redundancy = [g.avg_redundancy for g in SHAVED.guesses]
            plt.scatter(map_fitness, avg_redundancy, s=size)
            avg_redundancy_consistant = [g.avg_redundancy for g in SHAVED.guesses if g.consistant_key(key)]
            size_consistant = list(map(lambda x: (x - min(length))/(max(length)- min(length)) * 100 + 10, length_consistant))
            plt.scatter(map_fitness_consistant , avg_redundancy_consistant, s=size_consistant)
            plt.savefig("3-gram_map_fitness_avg_redundancy_shaved.jpg")

        SHAVED.print_guesses()

        if dev:
            pass
            # GRAPH STUFF
            # ==========
            #
            # plt.figure()
            # adj = SHAVED.get_adj_matrix()
            # G = nx.from_numpy_matrix(adj)

            # colors = []
            # for i in range(len(SHAVED.guesses)):
            #     if SHAVED.guesses[i].consistant_key(key):
            #         c = 'orange'
            #     else: c = 'blue'
            #     if SHAVED.consistancies[i] != 0:
            #         colors.append(c)

            # nx.draw(G, node_size=10, width=0.5, alpha=0.5, node_color=colors)
            # plt.axis('equal')
            # plt.savefig("consistancy_graph.jpg")

            # CONSISTANCY PLOTS
            # =================
            # plt.figure()
            # size = list(map(lambda x: (x - min(length))/(max(length)- min(length)) * 100 + 10, length))
            # consistancies = [i for i in SHAVED.consistancies]
            # plt.scatter(map_fitness, consistancies, s=size)
            # consistancies_consistant = [SHAVED.consistancies[i] for i in range(len(SHAVED.guesses)) \
            #                             if SHAVED.guesses[i].consistant_key(key)]
            # size_consistant = list(map(lambda x: (x - min(length))/(max(length)- min(length)) * 100 + 10, length_consistant))
            # plt.scatter(map_fitness_consistant , consistancies_consistant, s=size_consistant)
            # plt.savefig("3-gram_map_fitness_consistancies_shaved.jpg")

            # plt.figure()
            # size = list(map(lambda x: (x - min(length))/(max(length)- min(length)) * 100 + 10, length))
            # plt.scatter(avg_redundancy, consistancies, s=size)
            # size_consistant = list(map(lambda x: (x - min(length))/(max(length)- min(length)) * 100 + 10, length_consistant))
            # plt.scatter(avg_redundancy_consistant , consistancies_consistant, s=size_consistant)
            # plt.savefig("3-gram_avg_red_consistancies_shaved.jpg")

            # print()
            # guesses_for_print = sorted(SHAVED.guesses, key=lambda x: x.map_fitness)
            # for g in guesses_for_print:
            #     print(f"{g.gen_key()}  {g.ngram_fitness: .2f}  {g.avg_redundancy: .2f}  {g.map_fitness: .2f}  {g.consistant_key(key): 1}" )

    if(p['SEARCH_2']):
        C_DATA['2'] = get_multiplicities(get_Ngrams(cipher, 2))
        CRACK = Crack(SHAVED.guesses)
        CRACK.delete_guesses_forb_quads(cipher)

        CRACK.gen_guesses_ngram(2, True)
        CRACK.filter(0.5,0.9,1)
        CRACK.shrink()

        CRACK.get_maps_w_redundancy()
        CRACK.calc_map_fitness()
        CRACK.filter(1,1,0.4)
        CRACK.get_maps_w_redundancy()
        CRACK.guesses = sorted(CRACK.guesses, key=lambda x: -x.map_fitness)
        CRACK.print_guesses()

        if dev:
            plt.figure()
            fitness = [g.ngram_fitness for g in CRACK.guesses]
            map_fitness = [g.map_fitness for g in CRACK.guesses]
            plt.scatter(fitness, map_fitness)
            map_fitness_consistant = [g.map_fitness for g in CRACK.guesses if g.consistant_key(key)]
            fitness_consistant = [g.ngram_fitness for g in CRACK.guesses if g.consistant_key(key)]
            plt.scatter(fitness_consistant , map_fitness_consistant)
            plt.savefig("2-gram_fitness_map_fitness.jpg")

    if (SEARCH_W):
        start = time.perf_counter()
        print("=== WORDS ===")

        # CRACK.get_bad_maps_from_forb_quad()
        # CRACK.get_maps_w_redundancy()
        # print(len(CRACK.bad_maps), len(CRACK.maps))
        # for m in CRACK.maps:
        #     bad = False
        #     for bm in CRACK.bad_maps:
        #         if m[0]==bm[0] and m[1]==bm[1]:
        #             bad = True
        #             break
        #     if not bad:
        #         print(m)

        CRACK.delete_guesses_forb_quads(cipher)

        CRACK.shrink()
        CRACK.get_maps_w_redundancy()
        CRACK.calc_map_fitness()
        CRACK.print_guesses()

        print("First Search")
        if dev:
            CRACK = Crack([g for g in CRACK.guesses if g.consistant_key(key)])
        NEXT = Crack(CRACK.get_guesses_words(cipher, mode="start"))
        NEXT.word_filter(0.2, 0.2)
        NEXT.shrink()
        word_fitness = [g.words_fitness * g.avg_word_freq for g in NEXT.guesses]
        guesses = [x for _, x in sorted(zip(word_fitness, NEXT.guesses), key=lambda x: x[0])]
        NEXT = Crack(guesses)
        NEXT.print_guesses()
        NEXT.get_min()
        print("min:", NEXT.min.gen_key())
        print()

        for g in NEXT.guesses:
            print(g.route_words)

        print("Second Search (next_longest)")
        next_guesses = NEXT.get_guesses_words(cipher, mode="next_longest")
        next_guesses += NEXT.guesses # add old guesses to list
        NEXT_2 = Crack(next_guesses)
        NEXT_2.delete_guesses_forb_quads(cipher)
        NEXT_2.shrink()
        NEXT_2.print_guesses()
        NEXT_2.get_min()
        print("min:", NEXT_2.min.gen_key())
        print()

        print("Third Search (next_gap)")
        NEXT_3 = Crack(NEXT_2.get_guesses_words(cipher, mode="next_gap") + NEXT_2.guesses)
        NEXT_3.delete_guesses_forb_quads(cipher)
        NEXT_3.shrink()
        NEXT_3.get_maps_w_redundancy()
        NEXT_3.shave(0.8)
        NEXT_3.print_guesses()
        NEXT_3.get_min()
        print("min:", NEXT_3.min.gen_key())
        print()

        print("Fourth Search (next_gap)")
        NEXT_4 = Crack(NEXT_3.get_guesses_words(cipher, mode="next_gap") + NEXT_3.guesses)
        NEXT_4.delete_guesses_forb_quads(cipher)
        NEXT_4.shrink()
        NEXT_4.get_maps_w_redundancy()
        NEXT_4.shave(0.7)
        NEXT_4.print_guesses()
        NEXT_4.get_min()
        print("min:", NEXT_4.min.gen_key())
        print()

        if dev:
            pass
            # plt.figure()
            # fitness = [g.words_fitness for g in CRACK.guesses]
            # length = [len(g.dict) for g in CRACK.guesses]
            # plt.scatter(fitness, length)
            # fitness_consistant = [g.words_fitness for g in CRACK.guesses if g.consistant_key(key)]
            # length_consistant = [len(g.dict) for g in CRACK.guesses if g.consistant_key(key)]
            # plt.scatter(fitness_consistant , length_consistant)
            # plt.savefig("word_fitness_length.jpg")


    if(DECRYPT_GUESS):
        print("=== DECRYPT WITH GUESSES ===")
        if dev:
            try:
                print(NEXT_4.guesses[0].gen_key(), NEXT_4.guesses[0].consistant_key(key))
                print(decrypt(NEXT_4.guesses[0].gen_key(), cipher))
            except:
                print("NO GUESSES")
        else:
            print("decrypting with minimum guess")
            print(decrypt(NEXT_4.min.gen_key(), cipher))
            NEXT_4.save_guesses("out", 100)

    return True


if __name__ == "__main__":
    # key = 'KZVYTASCIFXORDNLPGJMBWQEHU'
    key = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # open message
    f = open("message.txt", "r")
    text = f.read().lower()
    # text = "This is a secret message. But it is short. over. \
    #        but if i try to make it longer, maybe then it can get cracked. Or i have to add more stuff to this\
    #        i can write even more. The news are bad. Maybe this wont work. the problem is clear. the length of this text is not sufficient\
    #        and i do not know what to write. The message is not clear."
    cipher = encrypt(key, text).replace(" ","")

    #cProfile.run('crack(parameter)')
    crack(cipher)
