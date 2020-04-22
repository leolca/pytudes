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
    def __init__(self, spelldic=None, distinctivefeaturesfile=None, weightObjFun=None):
        # call the parent constructor
        Spell.__init__(self, spelldic)
        self.dfeatures = self.load_distinctivefeatures(distinctivefeaturesfile)
        self.listOfPhones = list(self.dfeatures.phon)
        if weightObjFun is None:
           self.weightObjFun = (0.5, 0.5)
        else:
           if sum(weightObjFun) != 1:
              raise TypeError("Weights do not sum 1.")
           self.weightObjFun = weightObjFun

    def load_distinctivefeatures(self, dfile):
        import pandas as pd
        return pd.read_csv(dfile, encoding = 'utf8')
 
#    @classmethod 
#    def create_complete_dictionart_from_corpus(corpusfile):
#        $ cat big.txt | tr 'A-Z' 'a-z' | tr -sc 'A-Za-z' '\n' | sort | uniq -c | sort -n -r | head | awk '{print "echo "$1" "$2" $(echo "$2" | ./geteSpeakWordlist.sh -k -p)" }' | sh | column -t
 

########################################################################################################
# @property
# @blabla.setter


# probability smoothing

########################################################################################################

# soundex spell checker
