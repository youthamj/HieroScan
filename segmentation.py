import pickle
import re
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


def forwardwordsegmentation(sentences, dic):
    result =[]
    for sentence in sentences:
        if(len(sentence)):
            op1 = FMM(sentence, dic)
            op2 = tokenization(op1)
            result.append(transliterature(op2, dic))
    return result


with open("assets/segmentation_assets/dictionaryfinal.pickle", "rb") as fm:  # read
    dictionaryfinal = pickle.load(fm)

with open("assets/segmentation_assets/subword_dic_final.pickle", "rb") as fm:  # read
    subword_dic_final = pickle.load(fm)

with open("assets/segmentation_assets/last_word_dic.pickle", "rb") as f:  # write
    word_dic = pickle.load(f)

testsen = "R8Z1V30F34Z1I9F40G43"
# testing line for the function

#test0
# sentences = ['D21I9N35P8N29I9M44Z1F4X1N29D39', 'E34N35N35I9O31O31S29G1V31D21N35P98D35X1', 'V13N35D60O1Q1D21V31E34N35M20D58D1', 'N35Q3D28Z1Z1Y5D21V13F31V13', '']
#test1
# sentences = ['Aa15G26I9N35D156E23N29D156D156G7N35V31N35', 'N35D156D156N35W11N14M18Q3N35E23T30G5T30Y5T30', 'D58Z1V31Q3V31Y5T30G43T30Q3T30', 'D156V31V31I9V31M17Y5M17G5S29Y5D39V31T30Y5', 'V31G17G5D46G1N14N14D156Y5G5Y5', '']
#test2
# sentences = ['T30', 'N35V31M17I9E23S29V28O50G17N35M41V31', 'X1G39G25N35V31D156D21T30Y5M17X1', 'E23D21V31Z1N35X1V31G5G17D21N35', 'E23D156V31N14N14N14V31M41G5T30']
sentences = ['', 'E23Q3D21N1D21X1N18X1D21N37G43V28Z1D21M42', 'S29M20T30N35N35N35S29O4D21V13N35D60O4', 'O50G14D21I9D58Q3E34N35M17O4', 'G17Q7N35V13N35G25X1F35', 'M20Q3G1D4E23X1I9M17G43G43V4R8', 'D156D52N35V31D21P1D21G43E9N37P8N35Q3D156O1P98M17G43M42E9Y5M44V28G43H6E9M195M42N35Y5G7N35', 'V4G17O50W11D46E23N35M17Y3Z1']
print(forwardwordsegmentation(sentences, word_dic))
