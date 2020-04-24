# -*- coding: utf-8 -*-
import os
import re
import json
from collections import Counter
from math import log10

def exists(path):
    """Test whether a path exists. Returns False for broken symbolic links."""
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

def words(text): return re.findall(r"\b[a-zA-Z]+['-]?[a-zA-Z]*\b", text.lower())

def nlargest(d, n=None, thekey=None):
    """Get the n largest from a dict/list/tuple according to a given key."""
    if len(d) > 1:
       if n is None:
          n = len(d)
       if type(d) is dict:
          if thekey is None:
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
          elif thekey is not None:
             sd = sorted(d, key=thekey, reverse=True)
          else:
             sd = sorted(d, reverse=True)
          n = min(len(sd),n)
          return sd[:n]
    else:
       return d


class Spell:
    """Base spellchecker class"""
    def __init__(self, spelldic=None, corpus=None):
        """Initialize the spellchecker from as an empty Counter or load data from an existing Counter or from file, using load_WORDS method."""
        if type( spelldic ) is Counter:
           self.WORDS = spelldic
        elif type(spelldic) is str and exists(spelldic):
             self.WORDS = self.load_WORDS(spelldic)
        else:
           if corpus is not None:
              self.WORDS = self.load_WORDS_from_corpus(corpus)
           else:
              self.WORDS = Counter()
        self.N = self.get_corpus_length()
        self.M = self.get_max_frequency()
        self.m = self.get_min_frequency()
        self.language = 'en_US'

    @classmethod
    def from_file(cls, filename):
        return cls(filename)

    @classmethod
    def load_WORDS_from_corpus(cls, corpusfile):
        if exists(corpusfile):
            with open(corpusfile) as f:
                return Counter(words(f.read()))

    @classmethod
    def from_text_corpus(cls, textfile):
        """Create a Spell object from a text corpus."""
        wcounter = cls.load_WORDS_from_corpus(textfile)
        mySpell = cls(wcounter)
        mySpell.removefromdic( mySpell.createoddwordslist() )
        mySpell.N = mySpell.get_corpus_length()
        mySpell.M = mySpell.get_max_frequency()
        mySpell.m = mySpell.get_min_frequency()
        return mySpell

    def WORDS_len(self):
        return len(self.WORDS)

    def load_WORDS(self, filename):
        """Read data from a disk file. The file might be a json file or a list of counts and words, one per line, as a typical output of the bash command `uniq -c'."""
        if exists(filename):
           try:
              with open(filename) as f:
                 return Counter(json.load(f)) 
           except ValueError:
              WORDS = Counter()
              with open(filename) as f:
                   for line in f:
                       m = re.search(r'\s*(?P<count>\d+)?\s*(?P<word>\w+)', line)  	# (?P<name>regex) Named Capturing Groups and Backreferences  
											# https://www.regular-expressions.info/named.html
                       word = m.group('word')
                       if m.group('count') == None:
                          wcount = 0
                       else:
                          wcount = int( m.group('count') )
                       WORDS[word] = wcount
              return WORDS
           else:
              return None 
        else:
           return None
   
    def get_corpus_length(self):
        if self.WORDS is not None:
           return sum(self.WORDS.values())

    def get_max_frequency(self):
        return self.WORDS[max(self.WORDS, key=self.WORDS.get)]

    def get_min_frequency(self):
        return self.WORDS[min(self.WORDS, key=self.WORDS.get)]

    def get_hapaxlegomenon(self, n=None, acounter=None):
        """
        return a list of hapaxlegomenon
        if provided if an optional argument, return a list of n-legomenon (integer argumento) or [min,max]-legomenon (list argument with max and min)
        """
        if acounter is None:
           acounter = self.WORDS
        if n is None:
           n = [1, 1]
        if not isinstance(n, list):
           n = [1, n]
        return [w for w in acounter if acounter[w] < n[1]+1 and acounter[w] > n[0]-1]

    def removefromdic(self, klist):
        """
        remove a list of words from current speller dictionary
        """
        for key in klist:
            if key in self.WORDS:
               del self.WORDS[key]

    def createoddwordslist(self, n=5, checkEnchant=True):
        """
        create a list of odd words (might be used to remove them from the dictionaty)
        odd words are hapax(n)-legomenon (optionaly, which are not in enchant dictionary)
        """
        nlegomenonList = self.get_hapaxlegomenon(5)
        if checkEnchant:
           import enchant
           endic = enchant.Dict(self.language)
           oddList = []
           for r in nlegomenonList:
               if not endic.check(r):
                  oddList.append(r)
        else:
           oddList = nlegomenonList
        return oddList

    def P(self, word):
        """Return the MLE of a word's probability."""
        return float(self.WORDS[word]) / self.N

    def ObjectiveFunction(self, candidate, word):
        """Defines the mathematical computation of the function intended to maximize and therefore providing better solutions to the spellchecker.
        The basic class considers solely the probability of a given word.
        """
        return self.P(candidate)

    def correction(self, word, numcandidates=1):
        """Return the best spelling corrections for a given misspelled string.
        Uses an objective function to define what is considered a better solution.
        Args:
           word: The given misspelled word.
           numcandidates: Number of candidates to be retrieved (default=1).
        """
        corrcand = nlargest([c for c in self.candidates(word)], n=numcandidates, thekey=lambda candidates: self.ObjectiveFunction(candidates, word))
        if numcandidates == 1:
           if len(corrcand) > 0:
              return corrcand[0]
           else: return ''
        else:
           return corrcand
        #return max(self.candidates(word), key=self.P)

    def candidates(self, word):
        "Generate possible spelling corrections for word."
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def known(self, words):
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in self.WORDS)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

######################################################################################################## 

class KeyboardSpell(Spell):
    def __init__(self, spelldic=None, keyboardlayoutfile=None, weightObjFun=None):
        # call the parent constructor
        Spell.__init__(self, spelldic)
        #super(self.__class__, self).__init__(spelldic)
        # or Spell.__init__(self, dicFile)
        self.load_keyboard_layout(keyboardlayoutfile)
        self.set_weightObjFun(weightObjFun)
        #if weightObjFun is None:
        #   self.weightObjFun = (0.5, 0.5)
        #else:
        #   self.set_weightObjFun(weightObjFun)
           #if sum(weightObjFun) != 1:
           #   raise TypeError("Weights do not sum 1.")
           #self.weightObjFun = weightObjFun 

    def set_weightObjFun(self, weight):
        if weight is None:
            self.weightObjFun = (0.5, 0.5)
        else:
            if sum(weight) != 1:
                raise TypeError("Weights do not sum 1.")
            self.weightObjFun = weight

    def load_keyboard_layout(self, keyboardlayoutfile):
        """ 
        Read keyboard layout from JSON file or text file (in this case, performs a literal evaluation of the python string).
           Args:
              keyboardlayoutfile: A keyboard layout file in JSON format or using python syntax.
        """
        if keyboardlayoutfile is not None:
           if keyboardlayoutfile.endswith('.json'):
              with open(keyboardlayoutfile, 'r') as f:
                   self.kblayout = json.load(f)
           else:
              import ast
              with open(keyboardlayoutfile, 'r') as f:
                   self.kblayout = ast.literal_eval(f.read())

    def getCharacterCoord(self, c):
        """
        Finds a 2-tuple representing c's position on the given keyboard array.
        If the character is not in the given array, throws a ValueError
        """
        row = -1
        column = -1
        if self.kblayout is None:
            raise Exception("Speller keyboard is empty!")
        for kb in self.kblayout:
            for r in kb:
                if c in r:
                   row = kb.index(r)
                   column = r.index(c)
                   return (row, column)
        raise ValueError(c + " not found in given keyboard layout")

    def typoDistance(self, s, t, saturation=1000):
        """
        Finds the typo Manhattan distance (an integer) between two characters, based
        on the keyboard layout. The distance might be a saturated value.
        """
        # add one if one is lowercase and other is not (shift diff)
        addShiftDiff = int( s.islower() != t.islower() )
        sc = self.getCharacterCoord(s.lower())
        tc = self.getCharacterCoord(t.lower())
        return min( sum( [abs(x-y) for x,y in zip(sc,tc)] ) + addShiftDiff, saturation)

    def keyboard_damerau_levenshtein_distance(self, s1, s2, saturation=4):
        """
        Computes the Damerau-Levenshtein distance between two strings considering different typo distances according to their keyboard distance.
        The substitution cost is given by the keyboard distance between the two typos involved.
        The insertion and deletion cost is the minimum distance between the inserted/deleted typo and the previous and next typo.
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
                   cost = self.typoDistance(s1[i], s2[j], saturation=saturation)
                delcost = min( self.typoDistance(s1[i], s1[i-1], saturation=saturation) if i > 0 and i < lenstr1 else 10, 
                               self.typoDistance(s1[i], s1[i+1], saturation=saturation) if i > -1 and i < lenstr1-1 else 10
                             )
                inscost = min( self.typoDistance(s2[j], s2[j-1], saturation=saturation) if j > 0 and j < lenstr2 else 10,
                               self.typoDistance(s2[j], s2[j+1], saturation=saturation) if j > -1 and j < lenstr2-1 else 10
                             )
                #print 'delcost=' + str(delcost) + ', inscost=' + str(inscost) + ', cost=' + str(cost)
                d[(i,j)] = min(
                            d[(i-1,j)] + delcost, # deletion
                            d[(i,j-1)] + inscost, # insertion
                            d[(i-1,j-1)] + cost, # substitution
                           )
                if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                    d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition
        return d[lenstr1-1,lenstr2-1]

    def ObjectiveFunction(self, candidate, word, saturation=4):
        """
        Provides the objective function to the optimization process. 
        It balances the probability of a candidate and its typing keyboard distance from the misspelled word.

	        f
	   log ---
	        m           log d
	w0 ---------  - w1 ---------
	        M           log d   
	   log ---               max
	        m

	w_1 \frac{\log (f/m)}{\log (M/m)} - w_2 \frac{ \log d}{\log d_{max}} 
        """
        if self.weightObjFun[1] > 0:
           d = self.keyboard_damerau_levenshtein_distance(candidate, word, saturation)
           maxdist = saturation*max(len(candidate),len(word))
           if candidate in self.WORDS:
              return self.weightObjFun[0]*(log10(float(self.WORDS[candidate])/self.m) / log10(float(self.M)/self.m)) - self.weightObjFun[1]*(log10(float(d)) / log10(maxdist))
           else:
              return -d
           return Spell.ObjectiveFunction(self, candidate, word)
        else:
           return super(KeyboardSpell,self).ObjectiveFunction(candidate, word) 
        return self.P(candidate)

########################################################################################################

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
                cmd = "echo {} | ./geteSpeakWordlist.sh -p -k".format(w)
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

     def createSpellingDict(pdic, wDic=self.WORDS):
         """
         create spelling dictionary using the most frequent word (when two words have the same pronouncing)
         """
         sdic = {}
         for k, v in pdic.iteritems():
             if v not in sdic:
                sdic[v] = k
             elif wDic[k] > wDic[sdic[v]]:
                sdic[v] = k
         return sdic

    def getPhonesDataFrame(filename='phones_eqvs_table.txt'):
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
         normfeat = [len(df[column].unique()) for column in df]
         normfeat.pop(0)
         if ph1:
            f1 = df.loc[df['phon'] == ph1].transpose().ix[range(1,len(df.columns))].as_matrix()
            f1 = np.divide(f1, normfeat)
         if ph2:
            print ph2
            f2 = df.loc[df['phon'] == ph2].transpose().ix[range(1,len(df.columns))].as_matrix()
            print f2
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



#    @classmethod 
#    def create_complete_dictionart_from_corpus(corpusfile):
#        $ cat big.txt | tr 'A-Z' 'a-z' | tr -sc 'A-Za-z' '\n' | sort | uniq -c | sort -n -r | head | awk '{print "echo "$1" "$2" $(echo "$2" | ./geteSpeakWordlist.sh -k -p)" }' | sh | column -t
 

########################################################################################################
# @property
# @blabla.setter


# probability smoothing

########################################################################################################

# soundex spell checker
