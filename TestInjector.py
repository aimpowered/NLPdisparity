import re
import pandas as pd
import copy
import numpy as np
import evaluate
from docx import Document
import os
import unittest
from datasets import load_dataset
class TestInjector(unittest.TestCase):
    
    def setUp(self):
        dataset_wmt_enfr = load_dataset("wmt14",'fr-en', split='test')
        to_translate_wmt14_en = []
        for i in range(len(dataset_wmt_enfr)):
            to_translate_wmt14_en.append(dataset_wmt_enfr[i]['translation']['en'])
        wmt14_en = DataLoader(data=to_translate_wmt14_en, dataset_name="wmt14_en")
        #The files paths for homophones and confusing letters will probably have to change
        self.injector =  DyslexiaInjector(load=wmt14_en, homophone_path="data/homophones_dict.pickle",
                                        confusing_letters_path="data/confusing_letters_dict.pickle", 
                                        confusing_words_path="data/pedler_dict.pickle", seed=3)
    
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
        self.assertEqual(out_word, "Capitol")
    
    def test_confusing_letter_swapper(self):
        homophone_swapped = False
        confusing_word_swapped = False
        confusing_letter_swapped = False
        letters_swapped = 0
        original_word = "Shh"
        word = "shh"
        out_word, letters_swapped, confusing_letter_swapped = self.injector.confusing_letter_swapper(
            original_word, word, 1, 
            letters_swapped, homophone_swapped, 
            confusing_word_swapped, 
            confusing_letter_swapped)
        if letters_swapped > 0:
            self.assertNotEqual(out_word, word)
        #check that word follows the same capitalization as the original word
        swapped_count = 0
        for i in range(len(out_word)):
            if original_word.strip('".,?!:;()').strip("'")[i].isupper():
                self.assertEqual(out_word[i].isupper(), True)
            else:
                self.assertEqual(out_word[i].islower(), True)
            if out_word[i].lower() != word[i].lower():
                swapped_count += 1
        #checking if correct number of letters were swapped
        self.assertEqual(swapped_count, letters_swapped, f"actual_swapped_count: {swapped_count} | given_letters_swapped: {letters_swapped}")
        #poissbile outcomes based on current version of confusing_letters_dict.pickle July 18 2023
        self.assertRegex(out_word, "Shn|Snn|Chh|Chn|Cnh|Cnn")

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
        out_word = self.injector.insert_punctuation(original_word, word, punctuation, apostrophe, False, False)
        self.assertEqual(out_word, original_word)

    def get_sentence_stats(self, original_sentence, new_sentence, k):
        actual_homophones = 0
        actual_words_modified = 0
        actual_letters_swapped = 0
        actual_confusing_words_injected = 0
        for i in range(len(original_sentence)):
            homophone_swapped = False
            confusing_word_swapped = False
            if original_sentence[i].lower().strip('".,?!:;()').strip("'") != new_sentence[i].lower().strip('".,?!:;()').strip("'"):
                actual_words_modified += 1
                #check if if new_sentence is a homophone of original_sentence
                if original_sentence[i].lower().strip('".,?!:;()').strip("'") in self.injector.homophones_dict and new_sentence[i].lower().strip('".,?!:;()').strip("'") in self.injector.homophones_dict[original_sentence[i].lower().strip('".,?!:;()').strip("'")] and k == 0:
                    actual_homophones += 1
                    homophone_swapped = True
                #check if confusing word was injected
                if original_sentence[i].lower().strip('".,?!:;()').strip("'") in self.injector.confusing_words_dict and new_sentence[i].lower().strip('".,?!:;()').strip("'") in self.injector.confusing_words_dict[original_sentence[i].lower().strip('".,?!:;()').strip("'")] and k == 2:
                    actual_confusing_words_injected += 1
                    confusing_word_swapped = True
                if not homophone_swapped and not confusing_word_swapped and k == 1:
                    #get the acutal number of letters swapped
                    for j in range(len(original_sentence[i].strip('".,?!:;()').strip("'"))):
                        if original_sentence[i][j].lower() != new_sentence[i][j].lower():
                            if new_sentence[i][j].lower().strip('".,?!:;()').strip("'") in self.injector.confusing_letters_dict[original_sentence[i][j].lower().strip('".,?!:;()').strip("'")]:
                                actual_letters_swapped += 1
        return actual_homophones, actual_words_modified, actual_letters_swapped, actual_confusing_words_injected               
                
    def test_injector(self):
        all_sentences= ['I am a sentence.', 'Also, in a sentence, we can have different types of punctuatioN!', "Now that we've tested punctuation, let's test homophones.","Let's see what happens when he have an abreviation like NASA"]
        #test once with p_homophone = 1 and p_letter = 0 and p_confusing_word = 0, then test with p_homophone = 0 and p_letter = 1 and p_confusing_word = 0, then test with p_homophone = 0 and p_letter = 0 and p_confusing_word = 1
        for original_sentence in all_sentences:
            for i in range(3):
                if i == 0:
                    out_sentence, results = self.injector.injector(original_sentence, p_homophone=1, p_letter=0, p_confusing_word=0)
                if i == 1:
                    out_sentence, results = self.injector.injector(original_sentence, p_homophone=0, p_letter=1, p_confusing_word=0)
                if i == 2:
                    out_sentence, results = self.injector.injector(original_sentence, p_homophone=0, p_letter=0, p_confusing_word=1)
                self.assertNotEqual(out_sentence, original_sentence)
                #also test to make sure the correct amount of words were changed and number of homophones injected
                expected_homophones = results[0]
                expected_letters_swapped= results[1]
                expected_confusing_words_injected = results[2]
                expected_words_modified= results[3]
                actual_homophones, actual_words_modified, actual_letters_swapped, actual_confusing_words_injected = self.get_sentence_stats(original_sentence.split(), out_sentence.split(), k=i)
                print(f"original_sentence: {original_sentence} | new_sentence: {out_sentence}")
                self.assertEqual(actual_homophones, expected_homophones, f"actual_homophones: {actual_homophones} | expected_homophones: {expected_homophones}, i: {i}")
                self.assertEqual(actual_words_modified, expected_words_modified, f"actual_words_modified: {actual_words_modified} | expected_words_modified: {expected_words_modified}, i: {i}")
                self.assertEqual(actual_letters_swapped, expected_letters_swapped, f"actual_letters_swapped: {actual_letters_swapped} | expected_letters_swapped: {expected_letters_swapped}, i: {i}")
                self.assertEqual(actual_confusing_words_injected, expected_confusing_words_injected, f"actual_confusing_words_injected: {actual_confusing_words_injected} | expected_confusing_words_injected: {expected_confusing_words_injected}, i: {i}")
