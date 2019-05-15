import os
import json

file_name = 'malayalam_map_unique.csv'


with open(file_name, 'r', encoding='utf8') as f:
    lines = f.readlines()

lines.pop(0)

en_words = []
mal_words = []

for line in lines:
    sp = line.split(',')
    word_en = sp[1][1:-1]
    word_mal = ','.join(sp[2: -1])[1:-1]
    en_words.append(word_en)
    mal_words.append(word_mal)

en_tokens = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

mal_tokens = set()

for word in mal_words:
    for c in word:
        mal_tokens.add(ord(c))

mal_tokens = list(mal_tokens)

en_maxlen = max(map(len, en_words))
mal_maxlen = max(map(len, mal_words))

with open('en_tokens.json', 'w') as f:
    json.dump(en_tokens, f)


with open('mal_tokens.json', 'w') as f:
    json.dump(mal_tokens, f)

with open('en_maxlen.txt', 'w') as f:
    f.write(str(en_maxlen))

with open('mal_maxlen.txt', 'w') as f:
    f.write(str(mal_maxlen))
