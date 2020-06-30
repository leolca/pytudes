# -*- coding: utf-8 -*-
from .utils import exists, nlargest, removeMultiple
from .spell import Spell
from math import log10

class SoundexSpell(Spell):
    SOUNDEX_WORDS = {} #soundex dictionary
    groups = ["AEIOUHWY", "BFPV", "CGKJQSXZ", "DT", "L", "MN", "R"]
    #groups = ["AEIOUY", "BFPV", "CGKJQSXZ", "DT", "L", "MN", "R", "HW"] #NARA variation
    soundexlen = 0

    def __init__(self, spelldic=None, corpusfile=None, suffixfile=None, language=None, encoding=None, soundexfile=None, soundexlen=None, weightObjFun=None):
        # call the parent constructor
        Spell.__init__(self, spelldic, corpusfile, suffixfile, language, encoding)
        if soundexlen is None:
            self.soundexlen = 4
        else:
            self.soundexlen = soundexlen
        self.load_soundex_dictionary(soundexfile)
        self.set_weightObjFun(weightObjFun)

    @classmethod
    def from_file(cls, spelldic=None, corpusfile=None, suffixfile=None, language=None, encoding=None, soundexfile=None, soundexlen=None, weightObjFun=None):
        return cls(spelldic, corpusfile, suffixfile, language, encoding, soundexfile, soundexlen=None, weightObjFun=None)

    def load_soundex_dictionary(self, filename):
        import json
        if filename is not None and filename.endswith('.json') and exists(filename):
            with open(filename) as f:
                self.SOUNDEX_WORDS = json.load(f)
            a_key = next(iter(self.SOUNDEX_WORDS))
            the_sndx_length = len(self.SOUNDEX_WORDS[a_key])
            if the_sndx_length != self.soundexlen:
                import warnings
                warnings.warn("Incongruent Soundex length parameter was given. Updating it to {} (length found in the given dictionary).".format(the_sndx_length))
                self.soundexlen = the_sndx_length
        else:
            for w in self.WORDS:
                self.SOUNDEX_WORDS[w] = self.soundex(w, self.soundexlen)
            if filename is not None and filename.endswith('.json') and not exists(filename):
               with open(filename, 'w') as f:
                   json.dump(self.SOUNDEX_WORDS, f, indent=2)

    def soundex(self, s, mlen=None):
        # https://west-penwith.org.uk/misc/soundex.htm
        sndx = d = ""
        count = 1
        s = s.upper()
        if mlen is None:
            mlen = self.soundexlen
        for i in range(len(s)):
            if not sndx: # keep first letter
                sndx = s[i]
            else:
                if s[i] != s[i-1]: # remove dupplicates
                   for k in range(len(self.groups)):
                       if s[i] in self.groups[k]:
                           d = str(k)
                           continue
                   if d != "" and d != "0": # remove zeros
                       sndx += d
                       count += 1
            if count >= mlen:
                break
        #sndx = sndx.replace("0","")
        sndx += "0" * (mlen-len(sndx))
        return sndx

    def candidates(self, word):
        """Generate possible spelling corrections for word."""
        sndx = self.soundex(word)
        clist = [key for (key, value) in self.SOUNDEX_WORDS.items() if value == sndx]
        return self.known(clist)

    def set_weightObjFun(self, weight):
        if weight is None:
            self.weightObjFun = (0.5, 0.5)
        else:
            if sum(weight) != 1:
                raise TypeError("Weights do not sum 1.")
            self.weightObjFun = weight

    def ObjectiveFunction(self, candidate, word, weightObjFun=None):
        """
        Computes the objective function for a given pair of candidate and word.
        It balances between word frequency and edit distance.
        """
        if weightObjFun is None:
            weightObjFun = self.weightObjFun
        if weightObjFun[1] > 0:
           d = self.damerau_levenshtein_distance(candidate, word)
           maxdist = len(candidate) + len(word)
           if candidate in self.WORDS:
               return weightObjFun[0]*(log10(float(self.WORDS[candidate])/self.m) / log10(float(self.M)/self.m)) - weightObjFun[1]*(log10(float(d+1)) / log10(maxdist))
           else:
               return -d
        else:
           return super(SoundexSpell,self).ObjectiveFunction(candidate, word)
        return self.P(candidate)
