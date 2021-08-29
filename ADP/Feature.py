from collections import Counter
import math
import re
from math import log
import numpy as np
import pandas as pd
k = 5
def readFasta(fileName):
    with open(fileName) as f:
        mark, seq = '', ''
        for l in f:
            l = l.rstrip()
            if l.startswith(">"):
                if mark and seq:
                    yield mark.lstrip(">"), seq
                    mark, seq = '', ''
                mark = l
            else:
                seq += l
        if mark and seq:
            yield mark.lstrip(">"), seq

letters = list('ACDEFGHIKLMNPQRSTVWY')
header_AAC = []
header_AAC_full = []
for aa in letters:
    header_AAC_full.append(aa + '_full')
header_AAC.append(header_AAC_full)

def AAC(seq):
    length = float(len(seq))
    count = Counter(seq)
    seqAAC = [(count[aa] * 1.0 / length) for aa in letters]
    return seqAAC

header_BP = []
for i in range(1, k * 40 + 1):
    header_BP.append('BINARY_' + str(i))

def BP(seq):
    encodings = []
    for AA in seq:
        for AA1 in letters:
            tag = 1 if AA == AA1 else 0
            encodings.append(tag)
    return encodings


group1 = {'hydrophobicity_PRAM900101': 'RKEDQN', 'normwaalsvolume': 'GASTPDC', 'polarity': 'LIFWCMVY',
          'polarizability': 'GASDT', 'charge': 'KR', 'secondarystruct': 'EALMQKRH', 'solventaccess': 'ALFCGIVW'}
group2 = {'hydrophobicity_PRAM900101': 'GASTPHY', 'normwaalsvolume': 'NVEQIL', 'polarity': 'PATGS',
          'polarizability': 'CPNVEQIL', 'charge': 'ANCQGHILMFPSTWYV', 'secondarystruct': 'VIYCWFT',
          'solventaccess': 'RKQEND'}
group3 = {'hydrophobicity_PRAM900101': 'CLVIMFW', 'normwaalsvolume': 'MHKFRYW', 'polarity': 'HQRKNED',
          'polarizability': 'KMHFRYW', 'charge': 'DE', 'secondarystruct': 'GNPSD', 'solventaccess': 'MSPTHY'}
groups = [group1, group2, group3]
propertys = ('hydrophobicity_PRAM900101', 'normwaalsvolume', 'polarity', 'polarizability', 'charge', 'secondarystruct',
             'solventaccess')
header_CTD = []
header_CTD1 = []
header_CTD2 = []
header_CTD3 = []
for p in propertys:
    for g in range(1, len(groups) + 1):
        header_CTD1.append(p + '.G' + str(g))
    for tr in ('Tr1221', 'Tr1331', 'Tr2332'):
        header_CTD2.append(p + '.' + tr)
    for g in ('1', '2', '3'):
        for d in ['0', '25', '50', '75', '100']:
            header_CTD3.append(p + '.' + g + '.residue' + d)
header_CTD.append(header_CTD1 + header_CTD2 + header_CTD3)


def Count_C(sequence1, sequence2):
    sum = 0
    for aa in sequence1:
        sum = sum + sequence2.count(aa)
    return sum


def Count_D(aaSet, sequence):
    number = 0
    for aa in sequence:
        if aa in aaSet:
            number = number + 1
    cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
    cutoffNums = [i if i >= 1 else 1 for i in cutoffNums]
    code = []
    for cutoff in cutoffNums:
        myCount = 0
        for i in range(len(sequence)):
            if sequence[i] in aaSet:
                myCount += 1
                if myCount == cutoff:
                    code.append((i + 1) / len(sequence))
                    break
        if myCount == 0:
            code.append(0)
    return code

def CTD(seq):
    encodings = []
    code = []
    code2 = []
    CTD1 = []
    CTD2 = []
    CTD3 = []
    aaPair = [seq[j:j + 2] for j in range(len(seq) - 1)]
    for p in propertys:
        c1 = Count_C(group1[p], seq) / len(seq)
        c2 = Count_C(group2[p], seq) / len(seq)
        c3 = 1 - c1 - c2
        code = code + [c1, c2, c3]

        c1221, c1331, c2332 = 0, 0, 0
        for pair in aaPair:
            if (pair[0] in group1[p] and pair[1] in group2[p]) or (pair[0] in group2[p] and pair[1] in group1[p]):
                c1221 = c1221 + 1
                continue
            if (pair[0] in group1[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group1[p]):
                c1331 = c1331 + 1
                continue
            if (pair[0] in group2[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group2[p]):
                c2332 = c2332 + 1
        code2 = code2 + [c1221 / len(aaPair), c1331 / len(aaPair), c2332 / len(aaPair)]
        CTD1 = CTD1 + [value / float(len(seq)) for value in Count_D(group1[p], seq)]
        CTD2 = CTD2 + [value / float(len(seq)) for value in Count_D(group2[p], seq)]
        CTD3 = CTD3 + [value / float(len(seq)) for value in Count_D(group3[p], seq)]
    encodings.append(code + code2 + CTD1 + CTD2 + CTD3)
    return encodings

diPeptides = [aa1 + aa2 for aa1 in letters for aa2 in letters]


def DPC(seq):
    AADict = {}
    encodings = []
    for aa in range(len(letters)):
        AADict[letters[aa]] = aa
    tmpCode = [0] * 400
    for j in range(len(seq) - 2 + 1):
        tmpCode[AADict[seq[j]] * 20 + AADict[seq[j + 1]]] = tmpCode[AADict[seq[j]] * 20 + AADict[seq[j + 1]]] + 1
    if sum(tmpCode) != 0:
        tmpDPC = [i / sum(tmpCode) for i in tmpCode]
    encodings.append(tmpDPC)
    return encodings

group = {'alphatic': 'GAVLMI', 'aromatic': 'FYW', 'postivecharge': 'KRH', 'negativecharge': 'DE', 'uncharge': 'STCPNQ'}
groupKey = group.keys()
header_GAAC = []
header_GAAC_full = []
header_GAAC_NT = []
header_GAAC_CT = []
for i in groupKey:
    header_GAAC_full.append(i + '_full')
header_GAAC.append(header_GAAC_full)

def GAAC(seq):
    length = float(len(seq))
    count = Counter(seq)
    code = []
    myDict = {}
    for key in groupKey:
        for AA in group[key]:
            myDict[key] = myDict.get(key, 0) + count[AA]
        code.append(myDict[key] / length)
    return code

dipeptide = [g1 + '.' + g2 for g1 in groupKey for g2 in groupKey]

def GDPC(seq):
    index = {}
    myDict = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key
    for t in dipeptide:
        myDict[t] = 0
    sum = 0
    for j in range(len(seq) - 1):
        myDict[index[seq[j]] + '.' + index[seq[j + 1]]] = myDict[index[seq[j]] + '.' + index[seq[j + 1]]] + 1
        sum = sum + 1
    encodings = []
    code = []
    if sum == 0:
        for t in dipeptide:
            code.append(0)
    else:
        for t in dipeptide:
            code.append(myDict[t] / sum)
    encodings.append(code)
    return encodings

seqAAC, seqAAI, seqEntropy, seqGAAC = [], [], [], []
def out(file):
    seq_FNC = []
    for mark, seq in readFasta(file):
        seq3 = "%s" % seq[:k] + "%s" % seq[-k:]
        seq3 = seq3.ljust(k, " ")
        AAC_full = AAC(seq)
        seqAAC.append(AAC_full)
        seqBP = BP(seq3)
        seqCTD = CTD(seq)
        seqDPC = DPC(seq)
        seq_FNC.append([mark] + AAC_full  + seqBP + seqCTD[0] + seqDPC[0] )
    seq_FNC = pd.DataFrame(seq_FNC)
    seq_Feature = []
    seq_Feature.append(['class'] + header_AAC[0] + header_BP + header_CTD[0] + diPeptides )
    seq_Feature = pd.DataFrame(seq_Feature)
    Train = pd.concat([seq_Feature, seq_FNC], axis=0)
    Train = pd.DataFrame(Train)
    df = Train.iloc[1:, 0]
    TRAAC = Train.iloc[1:, 1:21].values
    TRBPNC = Train.iloc[1:, 21:221].values
    TRCTD = Train.iloc[1:, 221:368].values
    TRDPC = Train.iloc[1:, 368:768].values
    return  [df,TRAAC, TRBPNC, TRCTD, TRDPC]

