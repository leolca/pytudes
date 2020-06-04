# -*- coding: utf-8 -*-
import os
import re
import json
from collections import Counter
from collections import OrderedDict
from math import log10

def exists(path):
    """Test whether a path exists. Returns False for broken symbolic links."""
    try:
        st = os.stat(path)
    except os.error:
        return False
    return True

#def convert2unicode(s):
#    if type(s) == str:
#       return unicode(s, "utf-8")
#    else:
#       return s

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
    def __init__(self, spelldic=None, corpusfile=None, suffixfile=None):
        """Initialize the spellchecker from as an empty Counter or load data from an existing Counter or from file, using load_WORDS method."""
        self.corpusfile = corpusfile
        if type( spelldic ) is Counter:
            self.WORDS = spelldic
        elif type(spelldic) is str and exists(spelldic):
            self.WORDS = self.load_WORDS(spelldic)
        else:
           if self.corpusfile is not None:
              self.WORDS = self.load_WORDS_from_corpus(self.corpusfile)
           else:
              self.WORDS = Counter()
        self.N = self.get_corpus_length()
        self.M = self.get_max_frequency()
        self.m = self.get_min_frequency()
        if suffixfile:
            self.suffixes = loadSuffixes(suffixfile)
        else:
            self.suffixes = None
        try:
            import locale
            from icu import LocaleData
            self.language, self.encoding = locale.getdefaultlocale()
            data = LocaleData(self.language)
            self.alphabet = data.getExemplarSet()
        except ImportError:
            self.language = 'en_US'
            self.encoding = 'UTF-8'
            self.alphabet = 'abcdefghijklmnopqrstuvwxyz'

#    @classmethod
    def words(cls, text): 
        return re.findall(r"\w+(?:['-]\w+)*", text.lower())

    @classmethod
    def from_file(cls, spelldic=None, corpusfile=None):
        return cls(spelldic, corpusfile)

#    @classmethod
#    def from_dictionary(cls, spelldic):
#        if exists(spelldic):
#            myspell = cls(spelldic)
#            print("~~~~~ myspell created ~~~~")
#            return myspell

#    @classmethod
    def load_WORDS_from_corpus(cls, corpusfile):
        if exists(corpusfile):
            with open(corpusfile) as f:
                return Counter(cls.words(f.read()))

#    @classmethod
#    def from_text_corpus(cls, textfile):
#        """Create a Spell object from a text corpus."""
#        wcounter = cls.load_WORDS_from_corpus(textfile)
#        mySpell = cls(wcounter)
#        mySpell.removefromdic( mySpell.createoddwordslist() )
#        return mySpell

    def save_dictionary(self, dicfilename):
        with open(dicfilename, 'w') as f:
            json.dump(self.WORDS, f, indent=2)

    def WORDS_len(self):
        return len(self.WORDS)

    def load_WORDS(self, filename):
        """
        Read data from a disk file. 
        The file might be a json file or a list of counts and words, one per line, as a typical output of the bash command `uniq -c'.
        """
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
           n = (1, 1)
        if isinstance(n,int):
           n = (1, n)
        if isinstance(n, (tuple, list)) and len(n)==2:
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

    def loadSuffixes(self, sfxfile):
        if exists(sfxfile):
            try:
                sfxs = {}
                with open(sfxfile) as f:
                    for line in f:
                        li = line.strip()
                        li = re.sub(r' *#.*$','',li)
                        if li:
                            lf = li.split('\t')
                            sfxs[lf[0]] = lf[1]
                            for k in range(2,len(lf)):
                                sfxs[lf[0]] += ' ' + lf[k]
                suffixes = OrderedDict(sorted(sfxs.items(), key=lambda t: len(t[0]), reverse = True)) # suffixe replacement should be done by longest-match-first
                return suffixes
        else:
            return None

    def stripSuffix(self, word):
        if self.suffixes:
            wlist = []
            for regex in self.suffixes:
                m = re.search(regex, word)
                if m:
                    replace = self.suffixes[regex].split()
                    if not replace:
                        wlist = word[0:m.start()]
                    else:
                        for r in replace:
                            wlist.append(word[0:m.start()] + r)
                break
            return wlist
        return None

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
        if self.suffixes:
            return set(w for w in words if self.stripSuffix(w) in self.WORDS)
        else:
            return set(w for w in words if w in self.WORDS)

    def edits1(self, word):
        "All edits that are one edit away from `word`."
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in self.alphabet]
        inserts    = [L + c + R               for L, R in splits for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

