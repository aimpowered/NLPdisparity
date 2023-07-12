import unittest
from datasets import load_dataset
import DyslexiaInjector
class TestInjector(unittest.TestCase):
    
    def setUp(self):
        dataset_wmt_enfr = load_dataset("wmt14",'fr-en', split='test')
        to_translate_wmt14_en = []
        for i in range(len(dataset_wmt_enfr)):
            to_translate_wmt14_en.append(dataset_wmt_enfr[i]['translation']['en'])
        wmt14_en = DataLoader(data=to_translate_wmt14_en, dataset_name="wmt14_en")
        #The files paths for homophones and confusing letters will probably have to change
        self.injector =  DyslexiaInjector(load=wmt14_en, homophone_path="data/homophones_dict_v2.pickle", confusing_letters_path="data/confusing_letters_dict_v2.pickle", confusing_words_path="data/pedler_dict.pickle", seed=3)
    
    def test_get_punctuation(self):
        test_string = "Hello-W.o.rld's?"
        expected_punctuation = [(7, '.'), (9, '.'), (13, "'"), (15, '?')]
        self.assertEqual(self.injector.get_punctuation(test_string), expected_punctuation)
    
    def test_homophone_swapper(self):
        original_word = "Capital,"
        word = "capital"
        apostrophe = False
        out_word, apostrophe = self.injector.homophone_swapper(original_word, word)
        #check that first letter is capitalized
        self.assertEqual(out_word[0].isupper(), True)
        #check that words are different
        self.assertNotEqual(out_word, word)
    
    def test_confusing_letter_swapper(self):
        homophone_swapped = False
        confusing_word_swapped = False
        confusing_letter_swapped = False
        letters_swapped = 0
        original_word = "CapitAl,"
        word = "capital"
        out_word, letters_swapped, confusing_letter_swapped = self.injector.confusing_letter_swapper(original_word, word, 1, letters_swapped, homophone_swapped, confusing_word_swapped, confusing_letter_swapped)
        if letters_swapped > 0:
            self.assertNotEqual(out_word, word)
        #check that word follows the same capitalization as the original word
        for i in range(len(out_word)):
            if original_word.strip('".,?!:;()').strip("'")[i].isupper():
                self.assertEqual(out_word[i].isupper(), True)
            else:
                self.assertEqual(out_word[i].islower(), True)
    
    def test_confusing_word_injector(self):
        original_word = "Back"
        word = "back"
        out_word = self.injector.confusing_word_injector(original_word, word)
        print(f"original word: {original_word} | out_word: {out_word}")
        self.assertNotEqual(out_word, word)

    def test_insert_punctuation(self):
        original_word = "ca,pital."
        word = "capital"
        punctuation = [(2, ","), (8, '.')]
        apostrophe = True
        out_word = self.injector.insert_punctuation(original_word, word, punctuation, apostrophe)
        self.assertEqual(out_word, original_word)     