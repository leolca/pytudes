# -*- coding: utf-8 -*-
from spell.spell import *

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

