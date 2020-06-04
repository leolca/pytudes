# -*- coding: utf-8 -*-
from spell import *


class TypoSpell(Spell):
    def __init__(self, spelldic=None, corpusfile=None, weightObjFun=None):
        # call the parent constructor
        Spell.__init__(self, spelldic, corpusfile)
        self.compute_typo_statistics()
        self.set_weightObjFun(weightObjFun)

    @classmethod
    def from_file(cls, spelldic=None, corpusfile=None, weightObjFun=None):
        return cls(spelldic, corpusfile, weightObjFun)

    def set_weightObjFun(self, weight):
        if weight is None:
            self.weightObjFun = (0.5, 0.5)
        else:
            if sum(weight) != 1:
                raise TypeError("Weights do not sum 1.")
            self.weightObjFun = weight

    def compute_word_ngrams(self, word, n):
        from collections import deque
        ngram = deque(maxlen=n)
        ngrams_list = []
        word = '^' + word + '$'
        for c in word:
            ngram.append(c)
            if len(ngram) == n:
               ngrams_list.append(''.join(ngram))
        return ngrams_list

    def compute_typo_statistics(self):
        from collections import Counter
        self.BIGRAMS = Counter()
        self.TRIGRAMS = Counter()
        for w in self.WORDS:
            bigrams = self.compute_word_ngrams(w, 2)
            trigrams = self.compute_word_ngrams(w, 3)
            for bigram in bigrams:
                self.BIGRAMS[bigram] += self.WORDS[w]
            for trigram in trigrams:
                self.TRIGRAMS[trigram] += self.WORDS[w]
        #remove hapax legomenon / and-or use add one smoothing

    def trigram_peculiarity_index(self, trigram):
        import math
        bigram1 = trigram[0:2]
        bigram2 = trigram[1:3]
        if bigram1 in self.BIGRAMS and bigram2 in self.BIGRAMS and trigram in self.TRIGRAMS and self.BIGRAMS[bigram1] > 1 and self.BIGRAMS[bigram2] > 1 and self.TRIGRAMS[trigram] > 1:
            return (math.log10( self.BIGRAMS[bigram1] - 1 ) + math.log10( self.BIGRAMS[bigram2] - 1 ) )/2 - math.log10( self.TRIGRAMS[trigram] - 1 )
        else:
            return math.inf

    def word_peculiarity_index(self, word):
        trigrams = self.compute_word_ngrams(word, 3)
        r = 0
        for trigram in trigrams:
            r += self.trigram_peculiarity_index(trigram)
        return r/len(word)

    def ObjectiveFunction(self, candidate, word):
        if self.weightObjFun[1] > 0:
            ld = abs(len(candidate)-len(word)) + 1
            if candidate in self.WORDS:
                return -self.weightObjFun[0]*self.word_peculiarity_index(candidate)/ld -self.weightObjFun[1]*( log10(self.WORDS[candidate]-self.m+1) / log10(self.M)  )
            else:
                return -ld
        else:
            return super(TypoSpell,self).ObjectiveFunction(candidate, word)
