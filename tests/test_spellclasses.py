#!/usr/bin/python3
import unittest
import os, sys
from .context import spell
from functools import wraps
import logging
import logging.config
from datetime import datetime
from progressbar import ProgressBar

# make matplotlib silent
logging.basicConfig()
#logger = logging.getLogger(__name__)
logger = logging.getLogger('matplotlib')
logger.setLevel(logging.DEBUG)

# create logger
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_spellclasses.conf')
logging.config.fileConfig(log_file_path, defaults={'date':datetime.now().strftime("%Y%m%d-%H%M%S")})
logger = logging.getLogger('spell_unittest')
projpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

def my_logger(orig_func):
    """
    Decorator for unitest to log function calls
    """ 
    logger.info('function:{fname}'.format(fname=orig_func.__name__))

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logger.info(
            'Ran with args: {args}, and kwargs: {kwargs}'.format(args=args, kwargs=kwargs))
        return orig_func(*args, **kwargs)
    return wrapper

def my_timer(orig_func):
    """
    Decorator for unitest to get the time to execute a test
    """
    import time
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        print('{fname} ran in: {time} sec'.format(fname=orig_func.__name__, time=t2))
        return result
    return wrapper

class TestSpell(unittest.TestCase):
    # test data dictionaty from Roger Mitton's Home Page
    testdatasource =  {'birkbeck': 'https://www.dcs.bbk.ac.uk/~ROGER/missp.dat', 
                 'aspell': 'https://www.dcs.bbk.ac.uk/~ROGER/aspell.dat',
                 'wikipedia': 'https://www.dcs.bbk.ac.uk/~ROGER/wikipedia.dat',
                 'norvig1': 'https://norvig.com/spell-testset1.txt',
                 'norvig2': 'https://norvig.com/spell-testset2.txt',
                 'holbrook': 'https://www.dcs.bbk.ac.uk/~ROGER/holbrook-missp.dat'
                }
    #testdata = ['norvig1', 'norvig2'] #, 'aspell', 'wikipedia'] # select which dataset to use in tests
    testdata = ['wikipedia']
    N = None             # control how many tests will be performed (if None, tests everything)
    testDataSet = []
    testWeights = (0.5, 0.5) # freq, feature
    test_count = [0, 0] # count the number of tests (test_count[0]) and errors (test_count[1])
    positive = [0, 0]
    negative = [0, 0]
    Ncandidates = 7    # number of spell suggestion candidates
    test_count_candidates = [0] * Ncandidates
    Nodd = None         # remove odd words with frequency less or equal Nodd (if None, do not remove odd words)
    corpusfilename = projpath + "/data/big.txt"
    dictionaryfile = projpath + "/data/englishdict.json"
    sfxfile = None #projpath + "/data/english.sfx" # if None, do not use suffix strip
    kblayoutfile = projpath + "/data/qwertyKeymap.json" # use QWERTY as default keymap
    language = "en_US"
    encoding = "UTF-8"
    small_test_set =    (  ('speling', 'spelling'),   	# insert
                           ('korrectud','corrected'),      # replace 2
                           ('bycycle', 'bicycle'),         # replace
                           ('inconvient', 'inconvenient'), # insert 2
                           ('arrainged', 'arranged'),      # delete
                           ('peotry', 'poetry'),           # transpose
                           ('peotryy', 'poetry'),          # transpose + delete
                           ('word', 'word'),               # known
                           ('quintessential', 'quintessential'), # unknown
                         )

    def __init__(self, *args, **kwargs):
        super(TestSpell, self).__init__(*args, **kwargs)

    def setUp(self):
        """
        set up an array to collect assert failures 
        """
        print("+++ In method {} +++".format(self._testMethodName))
        self.verificationErrors = []

    def tearDown(self):
        """
        print all collected errors at tear down
        """
        for e in self.verificationErrors:
            logger.info(e)
        if self._testMethodName == 'test_suggestions':
            msg = "test: {testmethod} >>> {speller} has corrected {nerrors} from {total} spelling errors ({rate:.1%} correction rate)".format(
                testmethod=self._testMethodName, 
                speller=self.__class__.__name__, 
                nerrors=self.test_count[0]-self.test_count[1], 
                total=self.test_count[0],
                rate=float((self.test_count[0]-self.test_count[1])/self.test_count[0])
            )
            for i in range(self.Ncandidates):
                msg+='\nrate for {ncandidates} is {rate:.1%}'.format(
                        ncandidates=i+1,
                        rate=float((self.test_count[0]-self.test_count_candidates[i])/self.test_count[0])
                )
        elif self._testMethodName == 'test_isknown':
            msg = "test: {testmethod} >>> {speller} results:\n{truepos} true positives, {falsepos} false positives\n{trueneg} true negatives, {falseneg} false negatives".format(
                testmethod=self._testMethodName, 
                speller=self.__class__.__name__, 
		truepos=self.positive[0]-self.positive[1],
		falsepos=self.positive[1],
		trueneg=self.negative[0]-self.negative[1],
		falseneg=self.negative[1]
            ) 
        else:
            msg = 'nothing to do'
        logger.info(msg)
        print(msg)

    def Testset(self, lines):
        """
        Parse 'right: wrong1 wrong2' lines into [('right', 'wrong1'), ('right', 'wrong2')] pairs.
        """
        return [(right, wrong)
            for (right, wrongs) in (line.split(':') for line in lines)
            for wrong in wrongs.split()]

    def resetTestCount(self):
        self.test_count = [0, 0]
        self.positive = [0, 0]
        self.negative = [0, 0]

    def loadSpellFromCorpus(self, filename=None, sfxfile=None):
        if filename is None:
           filename = self.corpusfilename
        if sfxfile is None:
           sfxfile = self.sfxfile
        return spell.Spell.from_file(spelldic=None, corpusfile=filename, suffixfile=sfxfile)

    def loadSpellFromDictionary(self, filename=None, sfxfile=None):
        if filename is None:
           filename = self.dictionaryfile
        if sfxfile is None:
           sfxfile = self.sfxfile
        return spell.Spell.from_file(spelldic=filename, corpusfile=None, suffixfile=sfxfile)

    @my_logger
    @my_timer
    def test_create_spellchecker_from_corpus(self):
        myspell = self.loadSpellFromCorpus( )
        logger.info( "spellchecker created from {corpus}".format( corpus=self.corpusfilename ) )
        logger.info( "number of words: {n}".format( n=myspell.WORDS_len() ) )
        logger.info( "corpus length: {l}".format( l=myspell.get_corpus_length() ) )
        #self.resetTestCount()
        for test in self.small_test_set:
            self.test_count[0]+=1
            try: self.assertEqual(myspell.correction(test[0]), test[1], "the correct spelling is {correct}".format( correct=test[1] ))
            except AssertionError as e:
                self.verificationErrors.append(str(e))
                self.test_count[1]+=1
        del myspell

    @my_logger
    @my_timer
    def test_create_spellchecker_from_dictionary(self):
        myspell = self.loadSpellFromDictionary( ) 
        logger.info( "spellchecker created from {dic}".format( dic=self.dictionaryfile ) )
        logger.info( "number of words: {n}".format( n=myspell.WORDS_len() ) )
        logger.info( "corpus length: {l}".format( l=myspell.get_corpus_length() ) )
        #self.resetTestCount()
        for test in self.small_test_set:
            self.test_count[0]+=1
            try:
                self.assertEqual(myspell.correction(test[0]), test[1], "the correct spelling is {correct}".format( correct=test[1] ))
            except AssertionError as e:
                self.verificationErrors.append(str(e))
                self.test_count[1]+=1
        del myspell

    @my_logger
    @my_timer
    def download_testdata(self):
        """
        download all data and process to create a file with the following format (each word in a new line):
        word: misspelled1 misspelled2 
        """
        if len(self.testDataSet) == 0:
           logger.info("loading test data set ...")
           for key in self.testdata:
               if (not os.path.exists(projpath + "/data/" + key + ".dat")) and (not os.path.exists(projpath + "/data/spell-testset-" + key + ".txt")): 
                   print("downloading {}".format(key))
                   logger.info("downloading {}".format(key))
                   if self.testdatasource[key].endswith('.txt'):
                       os.system("wget -q " + self.testdatasource[key]  + " -O "+ projpath + "/data/spell-testset-"+ key + ".txt")
                   else:
                       os.system("wget -q " + self.testdatasource[key]  + " -O "+ projpath + "/data/"+ key + ".dat")
               else:
                   print("data for {key} found!".format(key=key))
               if not os.path.exists(projpath + "/data/spell-testset-" + key + ".txt"): 
                   print("converting...")
                   os.system("""cat """ + projpath + """/data/""" + key  + """.dat | tr -d '0-9' | tr '\n' ' ' | tr '$' '\n' | awk '{$1=$1":"}1' | sed '/^:$/d' > """ + projpath + """/data/spell-testset-""" + key + """.txt""")
               with open(projpath + "/data/spell-testset-" + key + ".txt") as f:
                   tmp = self.Testset( f )
                   print("{key} {length}".format(key=key,length=len(tmp)))
                   if self.N is not None:
                      import random
                      tmp = random.sample(tmp, self.N)
                   self.testDataSet.extend( set(tmp) - set(self.testDataSet) ) # extend list only with new entries
           logger.info("download completed!")

    @my_logger
    @my_timer
    def test_download_testdata(self):
        """
        test the list of words in the test datasets
        """
        self.download_testdata()
        myspell = spell.Spell.from_file(spelldic=None, corpusfile=self.corpusfilename, suffixfile=self.sfxfile)
        #self.resetTestCount()
        for right, wrong in self.testDataSet:
            with self.subTest(right=right):
                 self.test_count[0]+=1
                 #print("wrong word:{wrong}, right word:{right}".format(wrong=wrong, right=right))
                 try: self.assertEqual(myspell.correction(wrong), right, "the correct spelling is {correct}".format( correct=right ))
                 except AssertionError as e: 
                     self.verificationErrors.append(str(e))
                     self.test_count[1]+=1

    @my_logger
    @my_timer
    def test_isknown(self):
        """
        test if spellchecker identifies misspelled words and recognize correctly spelled words as right
        """
        self.download_testdata()
        logger.info("loading spellchecker ...")
        myspell = spell.Spell.from_file(spelldic=None, corpusfile=self.corpusfilename, suffixfile=self.sfxfile)
        if self.Nodd:
            myspell.removefromdic( myspell.createoddwordslist(n=self.Nodd) )
        logger.info("spellckecker ready!")
        # get unique (right, [wrongs])
        rights = set([item[0] for item in self.testDataSet])
        temp_testDataSet = []
        for r in rights:
            wrongs = [item[1] for item in self.testDataSet if item[0]==r]
            temp_testDataSet.append((r, wrongs))
        print("starting test - using {n} samples".format(n=len(temp_testDataSet)))
        pbar = ProgressBar()
        self.positive = [0, 0]
        self.negative = [0, 0]
        for right, wrongs in pbar(temp_testDataSet):
            self.test_count[0]+=1
            try:
                self.positive[0]+=1
                self.assertTrue(myspell.isknown(right))
            except AssertionError as e:
                self.verificationErrors.append(str(e) + ' right= ' + right)
                self.positive[1]+=1
            if type(wrongs) is str:
                wrongs = [wrongs]
            for wrong in wrongs:
                try:
                    self.negative[0]+=1
                    self.assertFalse(myspell.isknown(wrong))
                except AssertionError as e:
                    self.verificationErrors.append(str(e) + ' wrong= ' + wrong)
                    self.negative[1]+=1
        #print('true positive: {}\tfalse positive: {}'.format(positive[0]-positive[1],positive[1]))
        #print('true negative: {}\tfalse negative: {}'.format(negative[0]-negative[1],negative[1]))

    @my_logger
    @my_timer
    def test_suggestions(self):
        """
        test if the list of suggestions has the correct word
        """
        self.download_testdata()
        logger.info("loading spellchecker ...")
        myspell = spell.Spell.from_file(spelldic=None, corpusfile=self.corpusfilename, suffixfile=self.sfxfile)
        if self.Nodd:
            myspell.removefromdic( myspell.createoddwordslist(n=self.Nodd) )
        logger.info("spellckecker ready!")
        print("starting test - using {n} samples".format(n=len(self.testDataSet)))
        pbar = ProgressBar()
        for right, wrong in pbar(self.testDataSet):
        #for right, wrong in self.testDataSet:
            #with self.subTest(right=right):
                 self.test_count[0]+=1
                 candidates = myspell.correction(word=wrong, numcandidates=self.Ncandidates)
                 if type(candidates) is str:
                     candidates = [candidates]
                 try:
                     self.assertIn(right, candidates, "the correct spelling for {wrong} is {correct} (not listed as a sugestion)".format(wrong=wrong, correct=right))
                 except AssertionError as e:
                     if wrong in myspell.WORDS:
                         self.verificationErrors.append(str(e) + ' << real word error')
                     else:
                         self.verificationErrors.append(str(e))
                     self.test_count[1]+=1
                 for i in range(self.Ncandidates):
                     try:
                         self.assertIn(right, candidates[0:i+1], "the correct spelling for {wrong} is {correct} (not listed as a sugestion: {num})".format(wrong=wrong, correct=right, num=i+1))
                     except AssertionError as e:
                         self.verificationErrors.append(str(e))
                         self.test_count_candidates[i]+=1


class TestKeyboardSpell(TestSpell):
    #keyboardlayoutfile = 'qwertyKeymap.json'     # use QWERTY as default keymap

    def loadSpellFromCorpus(self, filename=None, suffixfile=None, keyboardlayoutfile=None, weightObjFun=None):
        if filename is None:
            filename = self.corpusfilename
        if suffixfile is None: 
            suffixfile = self.sfxfile
        if keyboardlayoutfile is None:
            keyboardlayoutfile = self.kblayoutfile
        if weightObjFun is None:
            weightObjFun = self.testWeights
        myspell = spell.KeyboardSpell.from_file(spelldic=None, corpusfile=filename, suffixfile=suffixfile, keyboardlayoutfile=keyboardlayoutfile, weightObjFun=weightObjFun)
        return myspell

    def loadSpellFromDictionary(self, filename=None, suffixfile=None, keyboardlayoutfile=None, weightObjFun=None):
        if filename is None:
            filename = projpath + "/data/englishdict.json"
        if suffixfile is None: 
           suffixfile = self.sfxfile
        if keyboardlayoutfile is None:
            keyboardlayoutfile = projpath + "/data/qwertyKeymap.json"
        if weightObjFun is None:
            weightObjFun = self.testWeights
        myspell = spell.KeyboardSpell.from_file(spelldic=filename, corpusfile=None, suffixfile=suffixfile, keyboardlayoutfile=keyboardlayoutfile, weightObjFun=weightObjFun)
        return myspell


class TestTypoSpell(TestSpell):

    def loadSpellFromCorpus(self, filename=None, suffixfile=None, weightObjFun=None):
        if filename is None:
            filename = self.corpusfilename
        if suffixfile is None:
            suffixfile = self.sfxfile
        if weightObjFun is None:
            weightObjFun = self.testWeights
        myspell = spell.KeyboardSpell.from_file(spelldic=None, corpusfile=filename, suffixfile=suffixfile, keyboardlayoutfile=keyboardlayoutfile, weightObjFun=weightObjFun)
        return myspell



class TestPhoneticSpell(TestSpell):

    def loadSpellFromCorpus(self, filename=None, suffixfile=None, pronalphabet=None, pronounciationdict=None, distinctivefeatures=None):
        if filename is None:
            filename = self.corpusfilename
        if suffixfile is None:
            suffixfile = self.sfxfile
        if pronalphabet is None:
            pronalphabet = 'ipa'
        if pronounciationdict is None:
            pronounciationdict = projpath+"/data/wordpronunciation.txt"
        if distinctivefeatures is None:
            distinctivefeatures=projpath+"/data/distinctivefeatures_ipa_mhulden.csv"
        myspell = spell.PhoneticSpell.from_file(spelldic=None, corpusfile=filename, suffixfile=suffixfile, pronalphabet=pronalphabet, pronounciationdict=pronounciationdict, distinctivefeatures=distinctivefeatures)
        #myspell.set_weight = (0.7, 0.3)
        return myspell

    def loadSpellFromDictionary(self, filename=None, suffixfile=None, ronalphabet=None, pronounciationdict=None, distinctivefeatures=None):
        if filename is None:
            filename = projpath + "/data/englishdict.json"
        if suffixfile is None:
            suffixfile = self.sfxfile
        if pronalphabet is None:
            pronalphabet = 'ipa'
        if pronounciationdict is None:
            pronounciationdict = projpath+"/data/wordpronunciation.txt"
        if distinctivefeatures is None:
            distinctivefeatures=projpath+"/data/distinctivefeatures_ipa_mhulden.csv"
        myspell = spell.PhoneticSpell.from_file(spelldic=filename, corpusfile=None, suffixfile=suffixfile, pronalphabet=pronalphabet, pronounciationdict=pronounciationdict, distinctivefeatures=distinctivefeatures)
        return myspell


def main():
    unittest.main()

if __name__ == '__main__':
    main()
