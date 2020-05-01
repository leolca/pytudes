# -*- coding: utf-8 -*-
from spell.spell import *

class PhoneticSpell(Spell):
    def __init__(self, spelldic=None, distinctivefeatures=None, pronounciationdict=None, weightObjFun=None):
        # call the parent constructor
        Spell.__init__(self, spelldic)
        self.load_distinctivefeatures(distinctivefeatures)
        self.loadwordpronounciationdict(pronounciationdict)
        if weightObjFun is None:
           self.weightObjFun = (0.5, 0.5)
        else:
           if sum(weightObjFun) != 1:
              raise TypeError("Weights do not sum 1.")
           self.weightObjFun = weightObjFun

    def load_distinctivefeatures(self, dfile):
        if dfile is not None and exists(dfile):
           import pandas as pd
           self.dfeatures = pd.read_csv(dfile, encoding = 'utf8')
           self.listOfPhones = list(self.dfeatures.phon)
        else:
           self.dfeatures = None
           self.listOfPhones = None

    def loadwordpronounciationdict(self, filename, pron='ipa', N=None):
        """
        load (create if dictionary file is not found)
        uses eSpeaker to get the ipa and kirshenbaum pronounciation
        """
        import pandas as pd
        if filename is None or not exists(filename):
            import subprocess
            df = pd.DataFrame(columns=['word', 'ipa', 'kirshenbaum'])
            for w in self.WORDS:
                cmd = "echo {w} | ../scripts/geteSpeakWordlist.sh -p -k".format(w=w)
                out = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]
                try:
                    df = df.append( pd.Series(out.decode("utf-8").split(), index = df.columns), ignore_index = True ) 
                except ValueError:
                    pass
        else:
            df = pd.read_csv(filename, sep='\t', lineterminator='\n', encoding = 'utf8')
        mdict = {}
        if pron == 'ipa':
           zdic = zip(df.word, df.ipa)
           for z in zdic:
               if z[0] not in mdict:
                  mdict[z[0]] = z[1]
        elif pron == 'kirshenbaum':
           zdic = zip(dfpd.word, dfpd.kirshenbaum)
           for z in zdic:
               if z[0] not in mdict:
                  mdict[z[0]] = z[1]
        else:
           raise NameError('wrong pronouncing dictionary')
        self.pronouncingDict = mdict

    @classmethod
    def from_text_corpus(cls, textfile=None, pron='ipa', pronounciationdict=None, distinctivefeatures=None):
        mySpell = super().from_text_corpus(textfile)
        mySpell.load_distinctivefeatures(distinctivefeatures)
        mySpell.loadwordpronounciationdict(pronounciationdict)
        return mySpell

    def parseUndesirableSymbols(words, undesirableSymbols = ["\xcb\x88", "\xcb\x90", "\xcb\x8c", "'", ",", "\xcb\x90", ":", ";", "2", "-"], replaceSymbols = [('ɜː','ɝ'), ('3:','R')]):
        if type(words) is dict:
            for w, value in words.items():
                rw = parseUndesirableSymbols(value, undesirableSymbols, replaceSymbols)
                words[w] = value
            return words
        elif type(words) is str:
            for c in undesirableSymbols:
                words = words.replace(c, "")
            for c,r in replaceSymbols:
                words = words.replace(c, r)
            return words
        else:
            return words

    def createSpellingDict(pdic, wDic=None):
        """
        create spelling dictionary using the most frequent word (when two words have the same pronouncing)
        """
        if wDic is None:
            wDic = self.WORDS
        sdic = {}
        for k, v in pdic.iteritems():
            if v not in sdic:
               sdic[v] = k
            elif wDic[k] > wDic[sdic[v]]:
               sdic[v] = k
        return sdic

    def getPhonesDataFrame(filename='../data/phones_eqvs_table.txt'):
        df = pd.read_csv(filename, encoding = 'utf8')
        return df

    def findWordGivenPhonetic(phonetic, mdic):
        rl = []
        for k,v in mdic.iteritems():
            if v == phonetic:
               rl.append(k)
        return rl

    def getWordPhoneticTranscription(word, alphabet=None, mdic=None):
        if mdic is not None and word in mdic:
           return parseUndesirableSymbols(mdic[word]) 
        else:
           if alphabet is None or alphabet.lower() == 'kirshenbaum':
              txtalphabet = '-x'
           elif alphabet.lower() == 'ipa':
              txtalphabet = '--ipa'
           out = subprocess.Popen(['espeak', '-q', txtalphabet, word], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
           stdout,stderr = out.communicate()
           if re.search("(\\r|)\\n$", stdout):
              stdout = re.sub("(\\r|)\\n$", "", stdout)
           stdout = stdout.lstrip() 
           return parseUndesirableSymbols(stdout)

    def find_sub_list(sl,l):
        results=[]
        sll=len(sl)
        for ind in (i for i,e in enumerate(l) if e==sl[0]):
            if l[ind:ind+sll]==sl:
               results.append((ind,ind+sll-1))
        return results

    def convertPhoneticWord2PhoneticSequence(word, letters=None):
        """ split word in a sequence of phones 
            tʃ, dʒ (ipa) and tS, dZ (kirshenbaum) are considered single phones
        """
        if letters is None:
            letters = self.listOfPhones
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
        normfeat = [len(df[column].unique()) for column in df]
        normfeat.pop(0)
        if ph1:
           f1 = df.loc[df['phon'] == ph1].transpose().ix[range(1,len(df.columns))].as_matrix()
           f1 = np.divide(f1, normfeat)
        if ph2:
           f2 = df.loc[df['phon'] == ph2].transpose().ix[range(1,len(df.columns))].as_matrix()
           f2 = np.divide(f2, normfeat)
        if f1 is not None and f2 is not None:
           return hammingdistance(f1,f2)
        elif f1 is not None and f1.size > 0:
           return sum(abs(f1))[0]
        elif f2 is not None and f2.size > 0:
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

    def phoneedits1(word, letters=None):
        "All edits that are one edit away from `word`."
        if letters is None:
            letters = self.listOfPhones
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

    def damerau_levenshtein_distance(s1, s2):
        """
        Compute the Damerau-Levenshtein distance between two given
        strings (s1 and s2)
        """
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
        normfeat = [len(df[column].unique()) for column in df]
        normfeat.pop(0)
        if ph1:
           f1 = df.loc[df['phon'] == ph1].transpose().ix[range(1,len(df.columns))].as_matrix()
           f1 = np.divide(f1, normfeat)
        if ph2:
           f2 = df.loc[df['phon'] == ph2].transpose().ix[range(1,len(df.columns))].as_matrix()
           f2 = np.divide(f2, normfeat)
        if f1 is not None and f2 is not None:
           return hammingdistance(f1,f2)
        elif f1 is not None and f1.size > 0:
           return sum(abs(f1))[0]
        elif f2 is not None and f2.size > 0:
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

    def phoneticCorrection(word, numcandidates=1, weight=(0.5, 0.5), alphabet='kirshenbaum', mdic=None, sdic=None, df=None, letters=None):
        "Most probable spelling correction for word."
        if mdic is None:
            mdic = self.pronouncingDict
        if sdic is None:
            sdic = self.spellingDict
        if df is None:
            df = self.distinctivefeatures
        if letters is None:
            letters = self.listOfPhones
        sword = getWordPhoneticTranscription(word, alphabet, mdic)
        if numcandidates > 1:
           wmax = nlargest([c for c in phoneticCandidates(sword, sdic)], n=numcandidates, thekey=lambda candidates: ObjectiveFunction(candidates, sword, sdic, df, weight, letters))
           corr = [''.join(w) for w in wmax]
           r = [sdic[c] for c in corr if c in sdic]
           if len(r) > 0:
              return r
           else:
              return word
        else:
           wmax = max([c for c in phoneticCandidates(sword, sdic)], key=lambda candidates: ObjectiveFunction(candidates, sword, sdic, df, weight, letters))
           corr = ''.join(wmax)
           if corr in sdic:
              return sdic[corr]
           else:
              return word

    def ObjectiveFunction(tword, oword, sdic, df, weight=(0.5, 0.5), letters=None, N=None, mm=None, MM=None):
        """ 
        function to maximize: 
          maximize probability of the existing target word and minimize edit distance between target word and original `word`
          sdic is a spelling dictionary
          df is a distinctive feature matrix
        """
        if letters is None:
            letters = self.listOfPhones
        if sum(weight) != 1:
           raise TypeError("Weights do not sum 1.")
        if N is None:
            N = sum(self.WORDS.values())
        if mm is None:
            mm = self.WORDS[min(self.WORDS, key=self.WORDS.get)]
        if MM is None:
            MM = self.WORDS[max(self.WORDS, key=self.WORDS.get)]
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



#    @classmethod 
#    def create_complete_dictionart_from_corpus(corpusfile):
#        $ cat big.txt | tr 'A-Z' 'a-z' | tr -sc 'A-Za-z' '\n' | sort | uniq -c | sort -n -r | head | awk '{print "echo "$1" "$2" $(echo "$2" | ./geteSpeakWordlist.sh -k -p)" }' | sh | column -t
 

########################################################################################################
# @property
# @blabla.setter


# probability smoothing

########################################################################################################

# soundex spell checker
