# -*- coding: utf-8 -*-
import os
import re
import collections
from collections import Counter
import subprocess
import pandas as pd
import numpy as np
import signal
signal.signal(signal.SIGPIPE,signal.SIG_DFL)
import csv
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
from math import log10
import enchant
#import pdb # debug


# global weight to use in the spell correction (probability, phonetic distance)
weightPdist = (0.5, 0.5)

endic = enchant.Dict("en_US")

def exists(path):
    """Test whether a path exists.  Returns False for broken symbolic links"""
    try:
        st = os.stat(path)
    except os.error:
        return False
    return True

def convert2unicode(s):
    if type(s) == str:
       return unicode(s, "utf-8")
    else:
       return s

def nlargest(d, n=None, thekey=None):
    if len(d) > 1:
       if n is None:
          n = len(d)
       if type(d) is dict:
          if key is None:
             sd = sorted([(v, k) for k, v in d.items()], reverse=True)
          else:
             sd = sorted([(thekey(k), k) for k, v in d.items()], reverse=True)
          n = min(len(sd),n)
          return sd[:n]
       else:
          if type(d) is set:
             d = [i for i in d]
          if type(d[0]) is tuple and thekey is not None:
             sd = sorted(d, key=lambda x: thekey(x[0]), reverse=True)
          else:
             sd = sorted(d, reverse=True)
          n = min(len(sd),n)
          return sd[:n]
    else:
       return d

def words(text): return re.findall(r"\b[a-zA-Z]+['-]?[a-zA-Z]*\b", text.lower())

WORDS = Counter(words(open('big.txt').read()))

def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return float(WORDS[word]) / N

def hapaxlegomenon(acounter):
    return nlegomenon(acounter, 1)

def nlegomenon(acounter, n):
    return [w for w in acounter if acounter[w] == n]

def nmostlegomenon(acounter, n):
    return [w for w in acounter if acounter[w] < n+1]

def removefromdict(klist, mdic):
    for key in klist:
        if key in mdic:
            del mdic[key]

nlegomenonList = nmostlegomenon(WORDS, 5)

removeList = []
for r in nlegomenonList:
   if not endic.check(r):
      removeList.append(r)
      
removefromdict(removeList, WORDS)

minword = min(WORDS, key=WORDS.get)
maxword = max(WORDS, key=WORDS.get)

distinctivefeatures = pd.read_csv('distinctivefeatures_kirshenbaum_mhulden.csv', encoding = 'utf8')
listOfPhones = list(distinctivefeatures.phon)

def loadwordpronounciationdict(filename, pron="ipa", N=None):
    if not os.path.exists(filename):
       cmd = "cat big.txt | tr 'A-Z' 'a-z' | tr -sc 'A-Za-z' '\\n' | sort | uniq -c | sort -n -r | tr -d '0-9 ' "
       if N is not None:
          cmd += "| head -n " + str(N) + " "
       cmd += "| ./geteSpeakWordlist.sh -p -k > " + filename
       print cmd
       os.system(cmd)
       cmd = "sed -i '1s/^/word\\tipa\\tkirshenbaum\\n/' " + filename
       print cmd
       os.system(cmd)
    dfpd = pd.read_csv(filename, sep='\t', lineterminator='\n', encoding = 'utf8')
    mdict = {}
    if pron == "ipa":
       zdic = zip(dfpd.word, dfpd.ipa)
       for z in zdic:
           if z[0] not in mdict:
               mdict[z[0]] = z[1]
    elif pron == "kirshenbaum":
       zdic = zip(dfpd.word, dfpd.kirshenbaum)
       for z in zdic:
           if z[0] not in mdict:
               mdict[z[0]] = z[1]
    else:
       raise NameError('wrong pronouncing dictionary')
    return mdict

# remove syllable and long vowel mark 
def parseUndesirableSymbols(mdic, undesirable=["\xcb\x88", "\xcb\x90", "\xcb\x8c", "'", ",", "\xcb\x90", ":", ";", "2", "-"]):
    if type(mdic).__name__ == 'dict':
       for key, value in mdic.items():
           if type(value) is unicode:
             value = value.replace(u'ɜː',u'ɝ')
             value = value.replace('3:','R')
           elif type(value) is str:
             value = value.replace('3:','R')
           for c in undesirable:
               value = value.replace(c, "")
           mdic[key] = value
       return mdic
    elif type(mdic).__name__ == 'str':
         mdic = mdic.replace('3:','R')
         for c in undesirable:
             mdic = mdic.replace(c, "")
         return mdic
    elif type(mdic).__name__ == 'unicode':
         mdic = mdic.replace('3:','R')
         mdic = mdic.replace(u'ɜː',u'ɝ')
         undesirableu = [convert2unicode(s) for s in undesirable]
         for c in undesirableu:
             mdic = mdic.replace(c, u"")
         return mdic

def getPhonesDataFrame(filename):
    df = pd.read_csv(filename, encoding = 'utf8')
    return df

def createSpellingDict(pdic, wDic=WORDS):
    sdic = {}
    for k, v in pdic.iteritems():
        if v not in sdic:
           sdic[v] = k
        elif wDic[k] > wDic[sdic[v]]:
           sdic[v] = k
    return sdic       

pronouncingDict = loadwordpronounciationdict('wordpronounciation.txt', 'kirshenbaum')
pronouncingDict = parseUndesirableSymbols(pronouncingDict)
removefromdict(removeList, pronouncingDict)
spellingDict = createSpellingDict(pronouncingDict)
df = getPhonesDataFrame('phones_eqvs_table.txt')

def findCollisions(mdic):
    w = collections.defaultdict(list)
    for k,v in mdic.iteritems():
        w[v].append(k)
    return [l for l in w.itervalues() if len(l)>1]

def findWordGivenPhonetic(phonetic, mdic):
    rl = []
    for k,v in mdic.iteritems():
        if v == phonetic:
           rl.append(k)
    return rl

def getWordPhoneticTranscription(word, alphabet=None, mdic=None):
    if mdic is not None and word in mdic:
       return convert2unicode( parseUndesirableSymbols(mdic[word]) )
    else:
       if alphabet is None or alphabet.lower() == 'kirshenbaum':
          txtalphabet = '-x'
       elif alphabet.lower() == 'ipa':
          txtalphabet = '--ipa'
       out = subprocess.Popen(['espeak', '-q', txtalphabet, word], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
       stdout,stderr = out.communicate()
       if re.search("(\\r|)\\n$", stdout):
          stdout = re.sub("(\\r|)\\n$", "", stdout)
       stdout = convert2unicode( stdout.lstrip() )
       return parseUndesirableSymbols(stdout)

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
           results.append((ind,ind+sll-1))
    return results

def convertPhoneticWord2PhoneticSequence(word, letters=listOfPhones):
    """ split word in a sequence of phones 
        tʃ, dʒ (ipa) and tS, dZ (kirshenbaum) are considered single phones
    """
    dlList = [l for l in listOfPhones if len(l) > 1]
    sequence = [c for c in word]
    for dl in dlList:
        sl = find_sub_list(list(dl), sequence)
        for k in range(len(sl)):
            s = sl[k]
            sequence[s[0]] = dl
            for x in range(s[0]+1,s[1]+1):
                sequence.pop(x)
                sl = [(i[0]-1, i[1]-1) for i in sl]
    return sequence

def convertPhoneticSequence2PhoneticWord(sword):
    if type(sword[0]) is list:
       wl = []
       for w in sword:
           wl.append(''.join(w))
       return wl
    else:
       return ''.join(sword)

def hammingdistance(x, y):
    return sum(abs(x - y))[0]

def phonedistance(ph1, ph2, df):
    f1 = f2 = None
    if ph1:
       f1 = df.loc[df['phon'] == ph1].transpose().ix[range(1,len(df.columns))].as_matrix()
    if ph2:
       f2 = df.loc[df['phon'] == ph2].transpose().ix[range(1,len(df.columns))].as_matrix()
    if f1 is not None and f2 is not None:
       return hammingdistance(f1,f2)
    elif f1 is not None:
       return sum(abs(f1))[0]
    elif f2 is not None:
       return sum(abs(f2))[0]
    else:
       return 0

def phoneticKnownWords(phoneword, mdic):
    "The subset of `words` that appear in the dictionary of WORDS."
    words = []
    for w in phoneword:
        possiblewords = findWordGivenPhonetic(w, mdic)
        if len(possiblewords) > 0:
           words.extend(possiblewords)
    return words

def phoneticKnown(phoneword, mdic):
    "The subset of `phonewords` that appear in the dictionary of WORDS."
    pwords = []
    for w in phoneword:
        if w in mdic:
           pwords.append(w)
    return pwords

def phoneticCandidates(pword, mdic):
    "Generate possible spelling corrections for word."
    return (phoneticKnown([pword], mdic) or phoneticKnown(phoneedits1(pword), mdic) or phoneticKnown(phoneedits2(pword), mdic) or [pword])

def phoneedits1(word, letters=listOfPhones):
    "All edits that are one edit away from `word`."
    word = convert2unicode(word)
    word = convertPhoneticWord2PhoneticSequence(word, letters)
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    deletes    = convertPhoneticSequence2PhoneticWord(deletes)
    transposes = [L + list(R[1]) + list(R[0]) + R[2:] for L, R in splits if len(R)>1]
    transposes = convertPhoneticSequence2PhoneticWord(transposes)
    replaces   = [L + list(c) + R[1:]           for L, R in splits if R for c in letters]
    replaces   = convertPhoneticSequence2PhoneticWord(replaces)
    inserts    = [L + list(c) + R               for L, R in splits for c in letters]
    inserts    = convertPhoneticSequence2PhoneticWord(inserts)
    return set(list(deletes) + list(transposes) + list(replaces) + list(inserts))

def phoneedits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in phoneedits1(word) for e2 in phoneedits1(e1))


"""
Compute the Damerau-Levenshtein distance between two given
strings (s1 and s2)
"""
def damerau_levenshtein_distance(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in xrange(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in xrange(-1,lenstr2+1):
        d[(-1,j)] = j+1
    for i in xrange(lenstr1):
        for j in xrange(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition
    return d[lenstr1-1,lenstr2-1]

def hammingdistance(x, y):
    return sum(abs(x - y))[0]

def phonedistance(ph1, ph2, df):
    f1 = f2 = None
    if ph1:
       f1 = df.loc[df['phon'] == ph1].transpose().ix[range(1,len(df.columns))].as_matrix()
    if ph2:
       f2 = df.loc[df['phon'] == ph2].transpose().ix[range(1,len(df.columns))].as_matrix()
    if f1 is not None and f2 is not None:
       return hammingdistance(f1,f2)
    elif f1 is not None:
       return sum(abs(f1))[0]
    elif f2 is not None:
       return sum(abs(f2))[0]
    else:
       return 0

def phoneDeleteDistance(s, df):
    if type(s) is list and (type(s[0]) is str or type(s[0]) is unicode) and len(s) == 2:
       w = 1
       if s[0] == '':
          w = 0.5 # weight
       return w * phonedistance(s[0], s[1], df)
    else:
       raise TypeError("Argument must be a list of strings (or unicodes) of length 2.")

def phoneInsertDistance(s, df):
    return phoneDeleteDistance(s, df)

def phoneReplaceDistance(s1, s2, df):
    if type(s1) is list and (type(s1[0]) is str or type(s1[0]) is unicode) and len(s1) == 2 and type(s2) is list and (type(s2[0]) is str or type(s2[0]) is unicode) and len(s2) == 2:
       if s1[1] == s2[1]:
          return 0
       w = 0.5 # weigh
       if s1[0] == '':
          w = 0.25 
       return phonedistance(s1[1], s2[1], df) + w * abs(phonedistance(s1[0], s1[1], df) - phonedistance(s2[0], s2[1], df))
    else:
        raise TypeError("Arguments must be a list of strings (or unicodes) of length 2 and the first element must be the same.")

def phoneTransposeDistance(s1, s2, df):
    if type(s1) is list and (type(s1[0]) is str or type(s1[0]) is unicode) and len(s1) == 3 and type(s2) is list and (type(s2[0]) is str or type(s2[0]) is unicode) and len(s2) == 3:
        w = 1 # weight
        if s1[0] == '' and s2[0] == '': 
           w = 0.5 
        return w * (  abs( phonedistance(s1[0], s1[1], df) - phonedistance(s1[0], s1[2], df) ) + abs( phonedistance(s2[0], s2[1], df) - phonedistance(s2[0], s2[2], df) ) )
    else:
        raise TypeError("Arguments must be a list of strings (or unicodes) of length 3.")

def phone_damerau_levenshtein_distance(s1, s2, df, letters):
    if (type(s1) is str or type(s1) is unicode) and (type(s2) is str or type(s2) is unicode):
       s1 = convertPhoneticWord2PhoneticSequence(s1, letters)
       s2 = convertPhoneticWord2PhoneticSequence(s2, letters)
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    d[(-1,-1)] = 0
    for i in xrange(0,lenstr1):
        d[(i,-1)] = d[(i-1,-1)] + phoneInsertDistance(['',s1[i]], df)
    for j in xrange(0,lenstr2):
        d[(-1,j)] = d[(-1,j-1)] + phoneInsertDistance(['',s2[j]], df)
    d[(lenstr1,-1)] = d[(lenstr1-1,-1)] + 1
    d[(-1,lenstr2)] = d[(-1,lenstr2-1)] + 1
    for i in xrange(lenstr1):
        for j in xrange(lenstr2):
            if s1[i] == s2[j]:
               cost = 0
            else:
               cost = 0.3
            d[(i,j)] = min(
                           d[(i-1,j)] + phoneDeleteDistance(s1[i-1:i+1] if i > 0 else ['']+[s1[i]], df), # deletion
                           d[(i,j-1)] + phoneInsertDistance(s2[j-1:j+1] if j > 0 else ['']+[s2[j]], df), # insertion
                           d[(i-1,j-1)] + cost * phoneReplaceDistance(s1[i-1:i+1] if i > 0 else ['']+[s1[i]], s2[j-1:j+1] if j > 0 else ['']+[s2[j]], df) # substitution
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
               d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost * phoneTransposeDistance(s1[i-2:i+1] if i > 1 else ['']+s1[i-1:i+1], s2[j-2:j+1] if j > 1 else ['']+s2[j-1:j+1], df)) # transposition
    return d[lenstr1-1,lenstr2-1]


def phoneticCorrection(word, mdic=pronouncingDict, sdic=spellingDict, df=distinctivefeatures, alphabet='kirshenbaum', letters=listOfPhones, weight=(0.5, 0.5)): 
    "Most probable spelling correction for word."
    sword = getWordPhoneticTranscription(word, alphabet, mdic)
    wmax = max([c for c in phoneticCandidates(sword, sdic)], key=lambda candidates: ObjectiveFunction(candidates, sword, sdic, df, letters, weight))
    corr = ''.join(wmax)
    if corr in sdic:
       return sdic[corr]
    else:
       return word


def ObjectiveFunction(tword, oword, sdic, df, letters=listOfPhones, weight=(0.5, 0.5), N=sum(WORDS.values()), mm=WORDS[min(WORDS, key=WORDS.get)], MM=WORDS[max(WORDS, key=WORDS.get)]): 
    """ 
    function to maximize: 
      maximize probability of the existing target word and minimize edit distance between target word and original `word`
      sdic is a spelling dictionary
      df is a distinctive feature matrix
    """
    if sum(weight) != 1:
       raise TypeError("Weights do not sum 1.")
    numfeatures = df.shape[1]
    d = 0
    if weight[1] > 0:
       d = phone_damerau_levenshtein_distance(tword, oword, df, letters)
    if type(tword) is list:
       tword = convertPhoneticSequence2PhoneticWord(tword)
    if tword in sdic:
       word = sdic[tword]
       if word in WORDS:
          # there are 25 (numfeatures) distinctive features ... then using weight 2x25 = 50 on probability
          return weight[0]*( log10(float(WORDS[word])/N) - log10(float(mm)/N) ) / (log10(float(MM)/N) - log10(float(mm)/N)) - weight[1]*float(d)/numfeatures
    elif weight[1] > 0:
       return -float(d)/numfeatures
    else:
       return 0


################ Test Code 
def unit_tests():
    w = weightPdist 
    assert phoneticCorrection('panks', weight=w) in ['banks', 'ranks']   # replace
    assert phoneticCorrection('dat', weight=w) in ['that', 'tat']        # replace
    assert phoneticCorrection('speling', weight=w) == 'spelling'         # insert
    assert phoneticCorrection('bycycle', weight=w) == 'bicycle'          # replace
    assert phoneticCorrection('inconvient', weight=w) == 'inconvenient'  # insert    
    assert phoneticCorrection('arrainged', weight=w) == 'arranged'       # delete
    assert phoneticCorrection('word', weight=w) == 'word'                # known
    assert phoneticCorrection('quintessential', weight=w) == 'quintessential' # unknown
    assert words('This is a TEST.') == ['this', 'is', 'a', 'test']
    assert Counter(words('This is a test. 123; A TEST this is.')) == (
           Counter({'this': 2, 'a': 2, 'is': 2, 'test': 2}))
    assert len(WORDS) == 26425
    assert sum(WORDS.values()) == 1079420
    assert WORDS.most_common(10) == [
     ('the', 79805),
     ('of', 40017),
     ('and', 38281),
     ('to', 28685),
     ('in', 21935),
     ('a', 21110),
     ('he', 12250),
     ('that', 12205),
     ('was', 11410),
     ('it', 10264)]
    assert WORDS['the'] == 79805
    assert P('quintessential') == 0
    assert 0.07 < P('the') < 0.08
    return 'unit_tests pass'

def spelltest(tests, verbose=False, log=False, weight=(0.5, 0.5)):
    "Run correction(wrong) on all (right, wrong) pairs; report results."
    import time
    if log:
       import datetime
       filename = 'phoneticspell_log_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
       f = open(filename,'w')
    start = time.clock()
    good, unknown = 0, 0
    n = len(tests)
    for right, wrong in tests:
        w = phoneticCorrection(wrong, weight=weight)
        good += (w == right)
        if w != right:
            unknown += (right not in WORDS)
            if verbose:
                print('correction({}) => {} ({}); expected {} ({})'
                      .format(wrong, w, WORDS[w], right, WORDS[right]))
            if log:
                print >>f, 'correction({}) => {} ({}); expected {} ({})'.format(wrong, w, WORDS[w], right, WORDS[right])
    if log:
       f.close()
    dt = time.clock() - start
    print('{:.0%} of {} correct ({:.0%} unknown) at {:.0f} words per second '
          .format(float(good) / n, n, float(unknown) / n, float(n) / dt))

def Testset(lines):
    "Parse 'right: wrong1 wrong2' lines into [('right', 'wrong1'), ('right', 'wrong2')] pairs."
    return [(right, wrong)
            for (right, wrongs) in (line.split(':') for line in lines)
            for wrong in wrongs.split()]

if __name__ == '__main__':
    print(unit_tests())
    w = weightPdist
    spelltest(Testset(open('spell-testset1.txt')), False, True, weight=w)
    spelltest(Testset(open('spell-testset2.txt')), False, True, weight=w)

