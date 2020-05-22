#!/usr/bin/python3
import unittest
import os, sys
from spell import Spell
from spell import KeyboardSpell
from spell import PhoneticSpell
from functools import wraps
import logging
import logging.config

# make matplotlib silent
logging.basicConfig()
#logger = logging.getLogger(__name__)
logger = logging.getLogger('matplotlib')
logger.setLevel(logging.DEBUG)

# create logger
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_spellclasses.conf')
logging.config.fileConfig(log_file_path)
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
    testdata =  {'birkbeck': 'https://www.dcs.bbk.ac.uk/~ROGER/missp.dat', 
                 'aspell': 'https://www.dcs.bbk.ac.uk/~ROGER/aspell.dat',
                 'wikipedia': 'https://www.dcs.bbk.ac.uk/~ROGER/wikipedia.dat'
                }
 
    N = 100             # control how many tests will be performed (if None, tests everything)
    testDataSet = []
    testWeights = (0.7, 0.3)
    test_count = [0, 0] # count the number of tests (test_count[0]) and errors (test_count[1])
    Ncandidates = 10    # number of spell suggestion candidates
    corpusfilename = projpath + "/data/big.txt"
    dictionaryfile = projpath + "/data/englishdict.json"
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
        msg = "test: {testmethod} >>> {speller} has corrected {nerrors} from {total} spelling errors ({rate:.1%} correction rate)".format(
                testmethod=self._testMethodName, 
                speller=self.__class__.__name__, 
                nerrors=self.test_count[0]-self.test_count[1], 
                total=self.test_count[0],
                rate=(self.test_count[0]-self.test_count[1])/self.test_count[0]
                )
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

    def loadSpellFromCorpus(self, filename):
        return Spell.from_file(spelldic=None, corpusfile=filename)

    def loadSpellFromDictionary(self, filename):
        return Spell.from_file(spelldic=filename, corpusfile=None)

    @my_logger
    @my_timer
    def test_create_spellchecker_from_corpus(self):
        myspell = self.loadSpellFromCorpus( self.corpusfilename )
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
        myspell = self.loadSpellFromDictionary( self.dictionaryfile )
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
               if not os.path.exists(projpath + "/data/" + key + ".dat"): 
                   logger.info("downloading {}".format(key))
                   os.system("wget -q " + self.testdata[key]  + " -O "+ projpath + "/data/"+ key + ".dat")
               os.system("""cat """ + projpath + """/data/""" + key  + """.dat | tr '\n' ' ' | tr '$' '\n' | awk '{$1=$1":"}1' | sed '/^:$/d' > """ + projpath + """/data/spell-testset-""" + key + """.txt""")
               with open(projpath + "/data/spell-testset-" + key + ".txt") as f:
                   tmp = self.Testset( f )
                   if self.N is not None:
                      import random
                      tmp = random.sample(tmp, self.N)
                   TestSpell.testDataSet.extend( set(tmp) - set(TestSpell.testDataSet) )
           logger.info("download completed!")

    @my_logger
    @my_timer
    def test_download_testdata(self):
        """
        test the list of words in the test datasets
        """
        self.download_testdata()
        myspell = Spell.from_file(spelldic=None, corpusfile=self.corpusfilename)
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
    def test_suggestions(self):
        """
        test if the list of suggestions has the correct word
        """
        self.download_testdata()
        myspell = Spell.from_file(spelldic=None, corpusfile=self.corpusfilename)
        for right, wrong in self.testDataSet:
            with self.subTest(right=right):
                 self.test_count[0]+=1
                 try: self.assertIn(right, myspell.correction(word=wrong, numcandidates=self.Ncandidates), "the correct spelling is {correct} (not listed as a sugestion)".format( correct=right ))
                 except AssertionError as e:
                     self.verificationErrors.append(str(e))
                     self.test_count[1]+=1


class TestKeyboardSpell(TestSpell):
    #keyboardlayoutfile = 'qwertyKeymap.json'     # use QWERTY as default keymap

    def loadSpellFromCorpus(self, filename=None, keyboardlayoutfile=None, weightObjFun=None):
        if filename is None:
            filename = self.corpusfilename
        if keyboardlayoutfile is None:
            keyboardlayoutfile = projpath + "/data/qwertyKeymap.json" # use QWERTY as default keymap
        if weightObjFun is None:
            weightObjFun = self.testWeights
        myspell = KeyboardSpell.from_file(spelldic=None, corpusfile=filename, keyboardlayoutfile=keyboardlayoutfile, weightObjFun=weightObjFun)
        return myspell

    def loadSpellFromDictionary(self, filename=None, keyboardlayoutfile=None, weightObjFun=None):
        if filename is None:
            filename = projpath + "/data/englishdict.json"
        if keyboardlayoutfile is None:
            keyboardlayoutfile = projpath + "/data/qwertyKeymap.json"
        if weightObjFun is None:
            weightObjFun = self.testWeights
        myspell = KeyboardSpell.from_file(spelldic=filename, corpusfile=None, keyboardlayoutfile=keyboardlayoutfile, weightObjFun=weightObjFun)
        return myspell


class TestPhoneticSpell(TestSpell):

    def loadSpellFromCorpus(self, filename=None, pronalphabet=None, pronounciationdict=None, distinctivefeatures=None):
        if filename is None:
            filename = self.corpusfilename
        if pronalphabet is None:
            pronalphabet = 'ipa'
        if pronounciationdict is None:
            pronounciationdict = projpath+"/data/wordpronunciation.txt"
        if distinctivefeatures is None:
            distinctivefeatures=projpath+"/data/distinctivefeatures_ipa_mhulden.csv"
        myspell = PhoneticSpell.from_file(spelldic=None, corpusfile=filename, pronalphabet=pronalphabet, pronounciationdict=pronounciationdict, distinctivefeatures=distinctivefeatures)
        #myspell.set_weight = (0.7, 0.3)
        return myspell

    def loadSpellFromDictionary(self, filename=None, pronalphabet=None, pronounciationdict=None, distinctivefeatures=None):
        if filename is None:
            filename = projpath + "/data/englishdict.json"
        if pronalphabet is None:
            pronalphabet = 'ipa'
        if pronounciationdict is None:
            pronounciationdict = projpath+"/data/wordpronunciation.txt"
        if distinctivefeatures is None:
            distinctivefeatures=projpath+"/data/distinctivefeatures_ipa_mhulden.csv"
        myspell = PhoneticSpell.from_file(spelldic=filename, corpusfile=None, pronalphabet=pronalphabet, pronounciationdict=pronounciationdict, distinctivefeatures=distinctivefeatures)
        return myspell


def main():
    unittest.main()

if __name__ == '__main__':
    main()
