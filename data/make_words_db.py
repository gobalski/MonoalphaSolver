#!/usr/bin/env python3
with open("words_370k.txt") as f:
    words = f.readlines()

words = [_[:-1] for _ in words]

letter_dicts = []

print("loading data..")
for word in words:
    if word in ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 't', 'u', 'v', 'w', 'x', 'z']:
        continue
    for idx, char in enumerate(word):
        if len(letter_dicts) - 1 < idx:
            letter_dicts.append({})
        if char in letter_dicts[idx]:
            letter_dicts[idx][char].add(word)
        else:
            letter_dicts[idx][char] = {word}

pattern = 'therelativelywiththenewhousesare'.replace('e','-')
print(f"find canditates for \"{pattern}\"")
sets = []
for idx, char in enumerate(pattern):
    if char != '-':
        try:
            sets.append(letter_dicts[idx][char])
        except KeyError:
            break
    else:
        sets.append(set(words))

print(set().union(*letter_dicts[0][:]))

candidates = set()
for idx, x in enumerate(sets):
    if idx + 1 == len(sets):
        partial_matches = sets[0].intersection(*sets)
        print(partial_matches)
    else:
        partial_matches = set([_ for _ in sets[0].intersection(*sets[0:idx + 1]) if len(_) == idx + 1])
        print(partial_matches)
    candidates = candidates.union(partial_matches)


print(candidates)

