import pickle
import pandas as pd
import string
from tqdm import tqdm
import re
import numpy as np
from difflib import SequenceMatcher


def tokenization(sentence):
    sen = []
    if len(sentence) == 1:
        sen.append(sentence[0])
        return sen
    code = ''
    count = 0
    index = 1
    i = 0
    for i in range(len(sentence) - 1):
        if sentence[i][0].isupper():
            while sentence[i + index].isdigit():
                index += 1
                count += 1
                if index >= 3:
                    break
                if (i + index) == len(sentence):
                    break
            if count == 0:
                sen.append(sentence[i])
                index = 1
                count = 0
            elif count == 1:
                code = sentence[i] + sentence[i + count]
                sen.append(code)
                index = 1
                count = 0
            elif count == 2:
                code = sentence[i] + sentence[i + count - 1] + sentence[i + count]
                sen.append(code)
                index = 1
                count = 0

    if sentence[i + 1].isupper():
        sen.append(sentence[i + 1])
    return sen


def cut_word(word):
    ind = [(m.start(0)) for m in re.finditer(r'[A-Z]', word)][-1]
    return word[:ind]


def one_glyph(word):
    try:
        ind = [(m.start(0)) for m in re.finditer(r'[A-Z]', word)][-2]
        return False
    except:
        return True


def FMM(sentence, dictA):
    result = []
    sentenceLen = len(sentence)
    maxDictA = max([len(word) for word in dictA])
    while sentenceLen > 0:
        word = sentence
        while word not in dictA:
            if one_glyph(word):
                break
            word = cut_word(word)
        result.append(word)
        sentence = sentence[len(word):]
        sentenceLen = len(sentence)
    return result


def transliterature(sen, dic):
    values = dic.values()
    values_list = list(values)
    finalop = []
    for j in range(len(sen)):
        if (dic.get(sen[j]) is not None):
            finalop.append(dic.get(sen[j])[0])
        else:
            try:
                similarties = list(
                    set(subword_dic_final.get(sen[j - 1])).intersection(set(subword_dic_final.get(sen[j]))))
                similarti_level = []
                for m in range(len(similarties)):
                    similarti_level.append(similar(dictionaryfinal["Gardiners"][similarties[m]], sen[j - 1] + sen[j]))
                max_score = max(similarti_level)
                q = similarti_level.index(max_score)
                finalop.append(dictionaryfinal["Transliteration"][similarties[q]])

            except:
                try:
                    similarties = list(
                        set(subword_dic_final.get(sen[j + 1])).intersection(set(subword_dic_final.get(sen[j]))))
                    similarti_level = []
                    for m in range(len(similarties)):
                        similarti_level.append(
                            similar(dictionaryfinal["Gardiners"][similarties[m]], sen[j + 1] + sen[j]))
                    max_score = max(similarti_level)
                    q = similarti_level.index(max_score)
                    finalop.append(dictionaryfinal["Transliteration"][similarties[q]])
                except:
                    finalop.append('')

    res = [''.join(charc) for charc in finalop]
    sentence = " ".join(res)
    if len(sentence) == 0:
        sentence = ''
    return sentence


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def forwardwordsegmentation(sentence, dic):
    op1 = FMM(sentence, dic)
    op2 = tokenization(op1)
    return transliterature(op2, dic)


with open("assets/segmentation_assets/dictionaryfinal.pickle", "rb") as fm:  # read
    dictionaryfinal = pickle.load(fm)

with open("assets/segmentation_assets/subword_dic_final.pickle", "rb") as fm:  # read
    subword_dic_final = pickle.load(fm)

with open("assets/segmentation_assets/last_word_dic.pickle", "rb") as f:  # write
    word_dic = pickle.load(f)

testsen = "R8Z1V30F34Z1I9F40G43"
# testing line for the function
print(forwardwordsegmentation(testsen, word_dic))
