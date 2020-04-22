#!/usr/bin/python3
import unittest
import os, sys
import spellclass as sc
from functools import wraps
import logging
import logging.config

# create logger
logging.config.fileConfig('spellclass_unittest.conf')
logger = logging.getLogger('spellclass_unittest')

def my_logger(orig_func):
    """
    Decorator for unitest to log function calls
    """ 
    logger.info('function:{}'.format(orig_func.__name__))

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        logger.info(
            'Ran with args: {}, and kwargs: {}'.format(args, kwargs))
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
        print('{} ran in: {} sec'.format(orig_func.__name__, t2))
        return result
    return wrapper


class TestSpell(unittest.TestCase):
   
    # test data dictionaty from Roger Mitton's Home Page
    testdata =  {'birkbeck': 'https://www.dcs.bbk.ac.uk/~ROGER/missp.dat', 
                 'aspell': 'https://www.dcs.bbk.ac.uk/~ROGER/aspell.dat',
                 'wikipedia': 'https://www.dcs.bbk.ac.uk/~ROGER/wikipedia.dat'
                }
 
    N = 10              # control how many tests will be performed (if None, tests everything)
    test_count = [0, 0] # count the number of tests (test_count[0]) and errors (test_count[1])

    def setUp(self):
        """
        set up an array to collect assert failures 
        """
        self.verificationErrors = []

    def tearDown(self):
        """
        print all collected errors at tear down
        """
        for e in self.verificationErrors:
            logger.info(e)
        logger.info('{} errors were found in {} cases'.format(str(self.test_count[1]), str(self.test_count[0])))

    def Testset(self, lines):
        """
        Parse 'right: wrong1 wrong2' lines into [('right', 'wrong1'), ('right', 'wrong2')] pairs.
        """
        return [(right, wrong)
            for (right, wrongs) in (line.split(':') for line in lines)
            for wrong in wrongs.split()]

    def loadSpellFromCorpus(self, filename):
        return sc.Spell.from_text_corpus( filename )

    @my_logger
    @my_timer
    def test_create_spellchecker_from_corpus(self):
        corpusfile = 'big.txt'
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
        myspell = self.loadSpellFromCorpus( corpusfile )
        logger.info( "spellchecker created from {}".format( corpusfile ) )
        logger.info( "number of words: {}".format( myspell.WORDS_len() ) )
        logger.info( "corpus length: {}".format( myspell.get_corpus_length() ) )
        for test in small_test_set:
            self.test_count[0]+=1
            try: self.assertEqual(myspell.correction(test[0]), test[1], "the correct spelling is {}".format( test[1] ))
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
        logger.info("downloading test data set ...")
        for key in self.testdata:
            if not os.path.exists(key + ".dat"): 
                os.system("wget -q " + self.testdata[key]  + " -O "+ key + ".dat")
            os.system("""cat """ + key  + """.dat | tr '\n' ' ' | tr '$' '\n' | awk '{$1=$1":"}1' | sed '/^:$/d' > spell-testset-""" + key + """.txt""")
        logger.info("download completed!")

    @my_logger
    @my_timer
    def test_download_testdata(self):
        """
        test the list of words in the test datasets
        """
        self.download_testdata()
        corpusfile = 'big.txt'
        myspell = sc.Spell.from_text_corpus( corpusfile )
        for key in self.testdata:
            with open("spell-testset-" + key + ".txt") as f:
                tests = self.Testset( f )
            if self.N is not None:
                import random
                tests = random.sample(tests, self.N)
            for right, wrong in tests:
                with self.subTest(right=right):
                    self.test_count[0]+=1
                    try: self.assertEqual(myspell.correction(wrong), right, "the correct spelling is {}".format( right ))
                    except AssertionError as e: 
                        self.verificationErrors.append(str(e))
                        self.test_count[1]+=1
            del tests

class TestKeyboardSpell(TestSpell):
    #keyboardlayoutfile = 'qwertyKeymap.json'     # use QWERTY as default keymap

    def loadSpellFromCorpus(self, filename=None, keyboardlayoutfile=None):
        myspell = sc.KeyboardSpell.from_text_corpus( filename )
        myspell.load_keyboard_layout('qwertyKeymap.json')   # use QWERTY as default keymap
        myspell.set_weight = (0.7, 0.3)
        return myspell

        #if filename is None:
        #    filename = 'englishdict.json'
        #if keyboardlayoutfile is None:
        #    keyboardlayoutfile = 'qwertyKeymap.json'     # use QWERTY as default keymap
        #return sc.KeyboardSpell(filename, keyboardlayoutfile, (0.7, 0.3))


def main():
    unittest.main()

if __name__ == '__main__':
    main()
