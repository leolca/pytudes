# -*- coding: utf-8 -*-
from spell.spell import *

__pronunciationalphabet__ = ('ipa', 'kirshenbaum')

class PhoneticSpell(Spell):
    def __init__(self, spelldic=None, corpusfile=None, pronalphabet=None, pronounciationdict=None, distinctivefeatures=None, weightObjFun=None):
        # call the parent constructor
        Spell.__init__(self, spelldic, corpusfile)
        if pronalphabet in __pronunciationalphabet__:
            self.pronalphabet = pronalphabet
        else:
            raise TypeError("Pronounciation alphabet invalid. Only the following are supported: " + ", ".join(__pronunciationalphabet__))
        self.loaddistinctivefeatures(distinctivefeatures)
        self.loadwordpronounciationdict(pronounciationdict, self.pronalphabet)
        self.spellingDict = self.createSpellingDict()
        if weightObjFun is None:
           self.weightObjFun = (0.5, 0.5)
        else:
           if sum(weightObjFun) != 1:
              raise TypeError("Weights do not sum 1.")
           self.weightObjFun = weightObjFun

    def loaddistinctivefeatures(self, dfile):
        if dfile is not None and exists(dfile):
            import pandas as pd
            self.distinctivefeatures = pd.read_csv(dfile, encoding = 'utf8')
            self.listOfPhones = list(self.distinctivefeatures.phon)
        else:
            raise TypeError("Distinctive features not found")
            #self.dfeatures = None
            #self.listOfPhones = None

    def loadwordpronounciationdict(self, filename, N=None):
        """
        load (create if dictionary file is not found)
        uses eSpeaker to get the ipa and kirshenbaum pronounciation
        """
        if self.pronalphabet is None:
            raise TypeError("A pronounciation alphabet must be choosen. Only the following are supported: " + ", ".join(__pronunciationalphabet__))
        import pandas as pd
        if filename is None or not exists(filename):
            import subprocess
            df = pd.DataFrame(columns=['word', 'ipa', 'kirshenbaum'])
            for w in self.WORDS:
                cmd = "echo {w} | ../scripts/geteSpeakWordlist.sh -p -k".format(w=w)
                out = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]
                out = str(out,'utf-8')
                try:
                    df = df.append( pd.Series(out.decode("utf-8").split(), index = df.columns), ignore_index = True ) 
                except ValueError:
                    pass
            if filename is not None:
                df.to_csv(filename, encoding='utf-8')
        else:
            df = pd.read_csv(filename, sep='\t', lineterminator='\n', encoding = 'utf8')
        mdict = {}
        if self.pronalphabet == 'ipa':
            zdic = zip(df.word, df.ipa)
            for z in zdic:
                if z[0] not in mdict:
                   mdict[z[0]] = self.parseUndesirableSymbols(z[1])
        elif self.pronalphabet == 'kirshenbaum':
            zdic = zip(dfpd.word, dfpd.kirshenbaum)
            for z in zdic:
                if z[0] not in mdict:
                   mdict[z[0]] = self.parseUndesirableSymbols(z[1])
        else:
            raise NameError('wrong pronouncing dictionary')
        self.pronouncingDict = mdict

#    @classmethod
#    def from_parent(cls, parent):
#        return cls(parent)

    @classmethod
    def from_file(cls, spelldic=None, corpusfile=None, pronalphabet=None, pronounciationdict=None, distinctivefeatures=None):
        return cls(spelldic, corpusfile, pronalphabet, pronounciationdict, distinctivefeatures)
        #mySpell = super().from_file(spelldic, corpusfile)
        #mySpell.loaddistinctivefeatures(distinctivefeatures)
        #mySpell.loadwordpronounciationdict(pronounciationdict)
        return mySpell

#    def from_dictionary(cls, spelldic, corpusfile=None, pronalphabet=None, pronounciationdict=None, distinctivefeatures=None, weightObjFun=None):
#        print("HERE I AM")
#        print(spelldic)
#        #mySpell = cls.from_parent( super().from_dictionary(spelldic) )
#        #mySpell = super().from_dictionary(spelldic, pron, distinctivefeatures, pronounciationdict, weightObjFun)
#        mySpell = PhoneticSpell(spelldic, pron, distinctivefeatures, pronounciationdict, weightObjFun)
#        #mySpell = super(PhoneticSpell, cls).__init__(spelldic=spelldic)
#        print("I AM HERE")
#        #mySpell.loaddistinctivefeatures(distinctivefeatures)
#        print("AMOST THERE")
#        #mySpell.loadwordpronounciationdict(pronounciationdict)
#        return mySpell

#    @classmethod
#    def from_text_corpus(cls, textfile=None, pron='ipa', pronounciationdict=None, distinctivefeatures=None, weightObjFun=None):
#        mySpell = super().from_text_corpus(textfile, pron, distinctivefeatures, pronounciationdict, weightObjFun)
#        #mySpell.loaddistinctivefeatures(distinctivefeatures)
#        #mySpell.loadwordpronounciationdict(pronounciationdict)
#        return mySpell

    def parseUndesirableSymbols(self, words, undesirableSymbols = ["\xcb\x88", "\xcb\x90", "\xcb\x8c", "ˈ", "'", ",", "ˌ", "\xcb\x90", "ː", ":", ";", "2", "-"], replaceSymbols = [('ɜː','ɝ'), ('3:','R')]):
        if type(words) is dict:
            for w, value in words.items():
                rw = self.parseUndesirableSymbols(value, undesirableSymbols, replaceSymbols)
                words[w] = value
            return words
        elif type(words) is list:
            for iw, w in enumerate(words):
                words[iw] = self.parseUndesirableSymbols(w, undesirableSymbols, replaceSymbols)
            return words
        elif type(words) is str:
            for c in undesirableSymbols:
                words = words.replace(c, "")
            for c,r in replaceSymbols:
                words = words.replace(c, r)
            return words
        else:
            return words

    def createSpellingDict(self, pdic=None, wDic=None):
        """
        create spelling dictionary using the most frequent word (when two words have the same pronouncing)
        """
        if pdic is None:
            pdic = self.pronouncingDict
        if wDic is None:
            wDic = self.WORDS
        sdic = {}
        for k, v in pdic.items():
            if v not in sdic:
               sdic[v] = k
            elif wDic[k] > wDic[sdic[v]]:
               sdic[v] = k
        return sdic

    def getPhonesDataFrame(self, filename='../data/phones_eqvs_table.txt'):
        df = pd.read_csv(filename, encoding = 'utf8')
        return df

    def findWordGivenPhonetic(self, phonetic, mdic):
        rl = []
        for k,v in mdic.iteritems():
            if v == phonetic:
               rl.append(k)
        return rl

    def getWordPhoneticTranscription(self, word, alphabet=None, mdic=None):
        if mdic is not None and word in mdic:
           return self.parseUndesirableSymbols(mdic[word]) 
        else:
           if alphabet is None or alphabet.lower() == 'kirshenbaum':
              txtalphabet = '-x'
           elif alphabet.lower() == 'ipa':
              txtalphabet = '--ipa'
           import subprocess
           out = subprocess.Popen(['espeak', '-q', txtalphabet, word], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
           stdout,stderr = out.communicate()
           stdout = str(stdout,'utf-8')
           if re.search("(\\r|)\\n$", stdout):
              stdout = re.sub("(\\r|)\\n$", "", stdout)
           stdout = stdout.lstrip() 
           return self.parseUndesirableSymbols(stdout)

    def find_sub_list(self, sl, l):
        """
        find a sublist in a give list
        returns a list ot tupples with start and end index of the sublist found in the given list
        """
        results=[]
        sll=len(sl)
        for ind in (i for i,e in enumerate(l) if e==sl[0]):
            if l[ind:ind+sll]==sl:
               results.append((ind,ind+sll-1))
        return results

    def convertPhoneticWord2PhoneticSequence(self, word, letters=None):
        """ 
        split word in a sequence of phones 
        in english, tʃ, dʒ (ipa) and tS, dZ (kirshenbaum) are considered single phones
        """
        # the spell checker list of phones will have them listed, so our convertion needs to mind
        # those symbols with length greater than 1
        if letters is None:
            letters = self.listOfPhones
        dlList = [l for l in letters if len(l) > 1]
        sequence = [c for c in word]
        for dl in dlList:
            sl = self.find_sub_list(list(dl), sequence) # find phonemes which are sequence-clusters (the output is a list of tupples with stand and end indexes)
            for k in range(len(sl)):
                s = sl[k]
                sequence[s[0]] = dl
                for x in range(s[0]+1,s[1]+1):
                    sequence.pop(x)
                    sl = [(i[0]-1, i[1]-1) for i in sl]
        return sequence

    def convertPhoneticSequence2PhoneticWord(self, sword):
        if type(sword[0]) is list:
           wl = []
           for w in sword:
               wl.append(''.join(w))
           return wl
        else:
           return ''.join(sword)

    def hammingdistance(self, x, y):
        return sum(abs(x - y))[0]

    def phoneticKnownWords(self, phoneword, mdic):
        "The subset of `words` that appear in the dictionary of WORDS."
        words = []
        for w in phoneword:
            possiblewords = self.findWordGivenPhonetic(w, mdic)
            if len(possiblewords) > 0:
               words.extend(possiblewords)
        return words

    def phoneticKnown(self, phoneword, mdic):
        "The subset of `phonewords` that appear in the dictionary of WORDS."
        pwords = []
        for w in phoneword:
            if w in mdic:
               pwords.append(w)
        return pwords

    def phoneticCandidates(self, pword, mdic):
        "Generate possible spelling corrections for word."
        return (self.phoneticKnown([pword], mdic) or self.phoneticKnown(self.phoneedits1(pword), mdic) or self.phoneticKnown(self.phoneedits2(pword), mdic) or [pword])

    def phoneedits1(self, word, letters=None):
        "All edits that are one edit away from `word`."
        if letters is None:
            letters = self.listOfPhones
        word = self.convertPhoneticWord2PhoneticSequence(word, letters)
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        deletes    = self.convertPhoneticSequence2PhoneticWord(deletes)
        transposes = [L + list(R[1]) + list(R[0]) + R[2:] for L, R in splits if len(R)>1]
        transposes = self.convertPhoneticSequence2PhoneticWord(transposes)
        replaces   = [L + list(c) + R[1:]           for L, R in splits if R for c in letters]
        replaces   = self.convertPhoneticSequence2PhoneticWord(replaces)
        inserts    = [L + list(c) + R               for L, R in splits for c in letters]
        inserts    = self.convertPhoneticSequence2PhoneticWord(inserts)
        return set(list(deletes) + list(transposes) + list(replaces) + list(inserts))

    def phoneedits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.phoneedits1(word) for e2 in self.phoneedits1(e1))

    def damerau_levenshtein_distance(self, s1, s2):
        """
        Compute the Damerau-Levenshtein distance between two given
        strings (s1 and s2)
        """
        d = {}
        lenstr1 = len(s1)
        lenstr2 = len(s2)
        for i in range(-1,lenstr1+1):
            d[(i,-1)] = i+1
        for j in range(-1,lenstr2+1):
            d[(-1,j)] = j+1
        for i in range(lenstr1):
            for j in range(lenstr2):
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

    def hammingdistance(self, x, y):
        return sum(abs(x - y))[0]

    def phonedistance(self, ph1, ph2, df):
        if ph1 == '':
            ph1 = None
        if ph2 == '':
            ph2 = None
        import numpy as np
        f1 = f2 = None
        normfeat = [len(df[column].unique()) for column in df]
        normfeat.pop(0)
        normfeat = np.array(normfeat)
        if ph1:
           f1 = df.loc[df['phon'] == ph1].transpose().iloc[range(1,len(df.columns))].values
           f1 = np.divide(f1, normfeat)
        if ph2:
           f2 = df.loc[df['phon'] == ph2].transpose().iloc[range(1,len(df.columns))].values
           f2 = np.divide(f2, normfeat)
        if f1 is not None and f2 is not None:
           return self.hammingdistance(f1,f2)
        elif f1 is not None and f1.size > 0:
           return sum(abs(f1))[0]
        elif f2 is not None and f2.size > 0:
           return sum(abs(f2))[0]
        else:
           return 0

    def phoneDeleteDistance(self, s, df):
        if type(s) is list and (type(s[0]) is str) and len(s) == 2:
           w = 1
           if s[0] == '' or s[1] == '':
              w = 0.5 # weight
           return w * self.phonedistance(s[0], s[1], df)
        else:
           raise TypeError("Argument must be a list of strings (or unicodes) of length 2.")

    def phoneInsertDistance(self, s, df):
        return self.phoneDeleteDistance(s, df)

    def phoneReplaceDistance(self, s1, s2, df):
        if type(s1) is list and (type(s1[0]) is str) and len(s1) == 2 and type(s2) is list and (type(s2[0]) is str) and len(s2) == 2:
           if s1[1] == s2[1]:
              return 0
           w = 0.5 # weigh
           if s1[0] == '':
              w = 0.25
           return self.phonedistance(s1[1], s2[1], df) + w * abs(self.phonedistance(s1[0], s1[1], df) - self.phonedistance(s2[0], s2[1], df))
        else:
           raise TypeError("Arguments must be a list of strings (or unicodes) of length 2 and the first element must be the same.")

    def phoneTransposeDistance(self, s1, s2, df):
        if type(s1) is list and (type(s1[0]) is str) and len(s1) == 3 and type(s2) is list and (type(s2[0]) is str) and len(s2) == 3:
            w = 1 # weight
            if s1[0] == '' and s2[0] == '':
               w = 0.5
            return w * (  abs( self.phonedistance(s1[0], s1[1], df) - self.phonedistance(s1[0], s1[2], df) ) + abs( self.phonedistance(s2[0], s2[1], df) - self.phonedistance(s2[0], s2[2], df) ) )
        else:
            raise TypeError("Arguments must be a list of strings (or unicodes) of length 3.")

    def phone_damerau_levenshtein_distance(self, s1, s2, df, letters):
        if (type(s1) is str) and (type(s2) is str):
           s1 = self.convertPhoneticWord2PhoneticSequence(s1, letters)
           s2 = self.convertPhoneticWord2PhoneticSequence(s2, letters)
        d = {}
        lenstr1 = len(s1)
        lenstr2 = len(s2)
        d[(-1,-1)] = 0
        for i in range(0,lenstr1):
            d[(i,-1)] = d[(i-1,-1)] + self.phoneInsertDistance(['',s1[i]], df)
        for j in range(0,lenstr2):
            d[(-1,j)] = d[(-1,j-1)] + self.phoneInsertDistance(['',s2[j]], df)
        d[(lenstr1,-1)] = d[(lenstr1-1,-1)] + 1
        d[(-1,lenstr2)] = d[(-1,lenstr2-1)] + 1
        for i in range(lenstr1):
            for j in range(lenstr2):
                if s1[i] == s2[j]:
                   cost = 0
                else:
                   cost = 0.3
                d[(i,j)] = min(
                          d[(i-1,j)] + self.phoneDeleteDistance(s1[i-1:i+1] if i > 0 else ['']+[s1[i]], df), # deletion
                          d[(i,j-1)] + self.phoneInsertDistance(s2[j-1:j+1] if j > 0 else ['']+[s2[j]], df), # insertion
                          d[(i-1,j-1)] + cost * self.phoneReplaceDistance(s1[i-1:i+1] if i > 0 else ['']+[s1[i]], s2[j-1:j+1] if j > 0 else ['']+[s2[j]], df) # substitution
                              )
                if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                   d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost * self.phoneTransposeDistance(s1[i-2:i+1] if i > 1 else ['']+s1[i-1:i+1], s2[j-2:j+1] if j > 1 else ['']+s2[j-1:j+1], df)) # transposition
        return d[lenstr1-1,lenstr2-1]

    #def phoneticCorrection(self, word, numcandidates=1, weight=(0.5, 0.5), alphabet='kirshenbaum', mdic=None, sdic=None, df=None, letters=None):
    def correction(self, word, numcandidates=1, weight=None, alphabet=None, mdic=None, sdic=None, df=None, letters=None):
        "Most probable spelling correction for word."
        if weight is None:
            weight = self.weightObjFun
        if alphabet is None:
            alphabet = self.pronalphabet
        if mdic is None:
            mdic = self.pronouncingDict
        if sdic is None:
            sdic = self.spellingDict
        if df is None:
            df = self.distinctivefeatures
        if letters is None:
            letters = self.listOfPhones
        sword = self.getWordPhoneticTranscription(word, alphabet, mdic)
        if numcandidates > 1:
           wmax = nlargest([c for c in self.phoneticCandidates(sword, sdic)], n=numcandidates, thekey=lambda candidates: self.ObjectiveFunction(candidates, sword, sdic, df, weight, letters))
           corr = [''.join(w) for w in wmax]
           r = [sdic[c] for c in corr if c in sdic]
           if len(r) > 0:
              return r
           else:
              return word
        else:
           wmax = max([c for c in self.phoneticCandidates(sword, sdic)], key=lambda candidates: self.ObjectiveFunction(candidates, sword, sdic, df, weight, letters))
           corr = ''.join(wmax)
           if corr in sdic:
              return sdic[corr]
           else:
              return word

    def ObjectiveFunction(self, tword, oword, sdic=None, df=None, weight=None, letters=None, N=None, mm=None, MM=None):
        """ 
        function to maximize: 
          maximize probability of the existing target word and minimize edit distance between target word and original `word`
          sdic is a spelling dictionary
          df is a distinctive feature matrix
        """
        if df is None:
            df = self.distinctivefeatures
        if weight is None:
            weight = self.weightObjFun
        if sdic is None:
            sdic = self.pronouncingDict
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
           #print("tword={tword},oword={oword}".format(tword=tword,oword=oword))
           d = self.phone_damerau_levenshtein_distance(tword, oword, df, letters)
        if type(tword) is list:
           tword = self.convertPhoneticSequence2PhoneticWord(tword)
        if tword in sdic:
           word = sdic[tword]
           if word in self.WORDS:
              # there are 25 (numfeatures) distinctive features ... then using weight 2x25 = 50 on probability
              return weight[0]*( log10(float(self.WORDS[word])/N) - log10(float(mm)/N) ) / (log10(float(MM)/N) - log10(float(mm)/N)) - weight[1]*float(d)/numfeatures
           else:
              return 0
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
