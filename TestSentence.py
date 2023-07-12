
import unittest
#objective of this test is to make sure that the sentences in the dataset are clean
class TestSentence(unittest.TestCase):
    def setUp(self):
        self.data = wmt14_en.get_data() #prob need to change this
    
    def test_for_xa0(self):
        for sentence in self.data:
            self.assertNotIn('xa0', sentence, sentence)
    
    def test_for_french_quotes(self):
        #testing related to french dataset
        for sentence in self.data:
            self.assertNotIn("«", sentence, sentence)
            self.assertNotIn("»", sentence, sentence)
    
    def test_for_newline(self):
        for sentence in self.data:
            self.assertNotIn("\n", sentence, sentence)
    
    def test_for_backslash(self):
        for sentence in self.data:
            self.assertNotIn("\\", sentence, sentence)
    
    def test_for_german_quotes(self):
        for sentence in self.data:
            self.assertNotIn("„", sentence, sentence)
            self.assertNotIn("“", sentence, sentence)

    def test_for_quote_number(self):
        for sentence in self.data:
            #make sure there is 0 or an even number of quotes
            self.assertEqual(sentence.count('"') % 2, 0, sentence)
