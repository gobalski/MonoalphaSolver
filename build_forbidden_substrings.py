#!/usr/bin/env python3

global ALPHABET
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

#open word data
def load_words():
    global ALPHABET

    with open("data/words_freq.txt") as f:
        words = f.readlines()
    words = [_[:-1].replace("'","") for _ in words]

    # remove words with special characters and apply lower case
    words = [word.lower() for word in words if all((l.upper() in ALPHABET) for l in word)]
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

    letter_dicts = []
    for word in words:
        for idx, char in enumerate(word):
            if len(letter_dicts) - 1 < idx:
                letter_dicts.append({})
            if char in letter_dicts[idx]:
                letter_dicts[idx][char].add(word)
            else:
                letter_dicts[idx][char] = {word}

    return letter_dicts, set(words), word_freq

#build all possible strings of lenth n
def all_strings(n):
    global ALPHABET
    if n == 1:
        return [l.lower() for l in ALPHABET]
    res_prev = all_strings(n-1)
    res = []
    for s in res_prev:
        for l in ALPHABET:
            l = l.lower()
            res.append(s+l)
    return res

def get_partition(n):
    pass

# construct allowed substrings of lentgh 4
def allowed_strings_4():
    letter_dicts, WORDS, word_freq = load_words()

    res = set()

    endings_2 = set()
    endings_3 = set()
    starts_2 = set()
    starts_3 = set()

    for word in WORDS:
        if len(word)>2:
            endings_2.add(word[-2:])
            endings_2.add(word[-2:])
            starts_3.add(word[-3:])
            starts_3.add(word[-3:])

    for word in WORDS:
        if len(word)>=4:
            for i in range(4,len(word)):
                res.add(word[i-4:i])
        elif len(word) == 2:
            for l1 in ALPHABET:
                for l2 in ALPHABET:
                    res.add(l1.lower() + word + l2.lower())
        elif len(word) == 1:
            for l in ALPHABET:
                for end in endings_2:
                    res.add(l.lower() + word + end)
                for start in starts_2:
                    res.add(start + word + l.lower())
        for end in endings_2:
            res.add(end + word[:2])
        for start in starts_2:
            res.add(word[-2:] + start)
        for end in endings_3:
            res.add(end + word[:1])
        for start in starts_3:
            res.add(word[-1:] + start)
        for l in ALPHABET:
            l = l.lower()
            res.add(word[-3:] + l)
            res.add(l + word[:3])

    return res

def save_forbidden_strings():
    all_str = set(all_strings(4))
    allowed_strings = allowed_strings_4()
    forbidden_str = all_str - allowed_strings
    with open("data/forbidden_quads.txt","w") as f:
        for line in forbidden_str:
            f.write(f"{line}\n")

save_forbidden_strings()
