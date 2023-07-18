import random
import pickle
import numpy as np
import DataLoader #Custom class
class DyslexiaInjector:
    """
    This class is used to inject dyslexia into a dataset. It can be used to inject homophones and confusing letters.
    ...
    Attributes
    ----------
    load : DataLoader
        The DataLoader object that contains the data that needs to be injected
    homophones_dict : dict
        A dictionary that contains the homophones
    confusing_letters_dict : dict
        A dictionary that contains the confusing letters
    Methods
    -------
    load_homophones(path)
        Loads the homophones from a pickle file
    load_confusing_letters(path)
        Loads the confusing letters from a pickle file
    injection_swap(p_start=0, p_end=1, step_size=0.1, save_path="", save_format="both")
        Injects dyslexia into the dataset by swapping words and letters.
    get_homophones(word)
        Returns the homophones of a word
    get_confusing_letters(letter)
        Returns the confusing letters of a letter
    injector(sentence, p_homophone, p_letter)
        Injects dyslexia into a sentence with a given probability
    Usage
    -------
    >>> from datasets import load_dataset
    >>> from DataLoader import DataLoader
    >>> from DyslexiaInjector import DyslexiaInjector
    >>> dataset_wmt_enfr = load_dataset("wmt14",'fr-en', split='test')
    >>> to_translate = []
    >>> for i in range(len(dataset_wmt_enfr)):
    >>>     to_translate.append(dataset_wmt_enfr[i]['translation']['en'])
    >>> loader = DataLoader(data=to_translate, dataset_name="wmt14_enfr")
    >>> dyslexia_injector = DyslexiaInjector(loader)
    >>> dyslexia_injector.injection_swap(p_start=0.1, p_end=0.5, step_size=0.1, save_path="data/wmt14_enfr", save_format="both")
    This creates multiple files with different levels of dyslexia injected into the dataset. The files are saved in the data folder.
    """
    def __init__(self, load: DataLoader, 
                homophone_path = "data/homophones_dict.pickle",
                confusing_letters_path = "data/confusing_letters_dict.pickle",
                confusing_words_path="data/pedler_dict.pickle",
                seed = 42):
        self.load = load
        self.homophones_dict = self.load_dict(homophone_path)
        self.confusing_letters_dict = self.load_dict(confusing_letters_path)
        self.confusing_words_dict = self.load_dict(confusing_words_path)
        random.seed(seed)

    def load_dict(self, path):
        with open(path, "rb") as f:
            out = pickle.load(f)
            #close the file
            f.close()
        return out   

    def injection_swap(self, p_start=0, p_end=1, step_size=0.1, save_path="", save_format="both"):
        """
        Injects dyslexia into the dataset by swapping words and letters. It is to note, that probability p does not result in p% of the words being modified.
        For example, if p = 0.5, it does not mean that 50% of the words will be modified. It means that each word has a 50% chance of being modified. But, not all words
        have homophones, confusing letters or consufing words. Therefore, the actual percentage of words that are modified is lower than p. The same applies to letters.
        Parameters
        ----------
        p_start : float
            The starting probability of swapping a word and letter
        p_end : float
            The ending probability of swapping a word and letter
        step_size : float
            The step size of the probability
        save_path : str
            The path where the data needs to be saved
        save_format : str
            The format in which the data needs to be saved. Can be "both", "csv" or "txt"
        """
        df_swap_results = pd.DataFrame(columns=["dataset","p_homophone", "p_letter", "p_confusing_word", "homophones_injected",
                                        "letters_swapped", "confusing_words_injected", "words_modified", "sentences_changed"])
        #for loop that increases the p_homophone with step_size
        for i in np.arange(p_start, p_end+step_size, step_size):
            #round i to 3 decimals
            i = round(i, 3)
            for j in np.arange(p_start, p_end+step_size, step_size):
                #round j to 3 decimals
                j = round(j, 3)
                for k in np.arange(p_start, p_end+step_size, step_size):
                    k = round(k, 3)
                    #create deep copy of the data
                    temp_load, results = self.injection_runner(self.load.create_deepcopy(), i, j, k)
                    df_swap_results.loc[len(df_swap_results)] = {"dataset":"wmt14_en", "p_homophone":i, "p_letter":j, "p_confusing_word":k,
                                                                "homophones_injected":results[0],"letters_swapped":results[1], 
                                                                "confusing_words_injected": results[2], "words_modified":results[3], 
                                                                "sentences_changed":results[4]}
                    self.saver(temp_load, save_path, i, j, k, save_format)
        #add number of sentences to the dataframe
        df_swap_results["sentences"] = self.load.get_number_of_sentences()
        #add number of words to the dataframe
        df_swap_results["words"] = self.load.get_number_of_words()
        #add number of characters to the dataframe
        df_swap_results["letters"] = self.load.get_number_of_letters()
        #percentage of sentences changed
        df_swap_results["percentage_sentences_changed"] = df_swap_results["sentences_changed"] / df_swap_results["sentences"] * 100
        #percentage of words modified
        df_swap_results["percentage_words_modified"] = df_swap_results["words_modified"] / df_swap_results["words"] * 100
        #percentaged of words swapped
        df_swap_results["percentage_words_swapped_for_homophones"] = df_swap_results["homophones_injected"] / df_swap_results["words"] * 100
        df_swap_results["percentage_words_swapped_for_confusing_words"] = df_swap_results["confusing_words_injected"] / df_swap_results["words"] * 100
        #percentage of letters swapped
        df_swap_results["percentage_letters_swapped"] = df_swap_results["letters_swapped"] / df_swap_results["letters"] * 100 
        #save the results
        df_swap_results.to_csv(f"{save_path}/swap_results.csv", index=False)
        return df_swap_results

    def injection_runner(self, data_loader, p_homophone, p_letter, p_confusing_word):
        #track of the amount of words that were swapped
        homophones_injected = 0
        #keep track of the amount of letters that were swapped
        letters_swapped = 0
        #keep track of the amount of confusing words that were injected
        confusing_words_injected = 0
        #track of the amount of words that were changed
        words_modified = 0
        #track of the amount of sentences that were changed
        sentences_changed = 0
        for i in range(len(data_loader.data)):
            #get the sentence
            sentence = data_loader.data[i]
            #swap the sentence
            sentence, results = self.injector(sentence, p_homophone, p_letter, p_confusing_word)
            # update the amount of words that were swapped
            homophones_injected += results[0]
            #update the amount of letters that were swapped
            letters_swapped += results[1]
            #updated the amount of confusing words that were injected
            confusing_words_injected += results[2]
            #update the amount of words that were changed
            words_modified += results[3]
            #update the amount of sentences that were changed
            if results[3] > 0:
                sentences_changed += 1
            #update the sentence in the dataframe
            data_loader.data[i] = sentence
        print(f"p_homophone = {p_homophone}, p_letter = {p_letter}, p_confusing_word = {p_confusing_word}")
        print("Homophones Injected: " + str(homophones_injected))
        print("Letters swapped: " + str(letters_swapped))
        print("Confusing words injected: " + str(confusing_words_injected))
        print("Words Modified: " + str(words_modified))
        print("Sentences changed: " + str(sentences_changed))
        return data_loader, (homophones_injected, letters_swapped, confusing_words_injected, words_modified, sentences_changed)

    def get_homophones(self, word):
        return self.homophones_dict[word]
    
    def get_homophone_dict(self):
        return self.homophones_dict
    
    def get_confusing_letters(self, word):
        return self.confusing_letters_dict[word]
    
    def get_confusing_letters_dict(self):
        return self.confusing_letters_dict
    
    def saver(self, temp_load: DataLoader, save_path, p_homophone, p_letter, p_confusing_word, format="both"):
        if format == "csv":
            temp_load.save_as_csv(save_path + f"{temp_load.get_name()}_p_homophone_{p_homophone}_p_letter_{p_letter}_p_confusing_word_{p_confusing_word}.csv")
        elif format == "txt":
            temp_load.save_as_txt(save_path + f"{temp_load.get_name()}_p_homophone_{p_homophone}_p_letter_{p_letter}_p_confusing_word_{p_confusing_word}.txt")
        else:
            #save as both
            temp_load.save_as_csv(save_path + f"{temp_load.get_name()}_p_homophone_{p_homophone}_p_letter_{p_letter}_p_confusing_word_{p_confusing_word}.csv")
            temp_load.save_as_txt(save_path + f"{temp_load.get_name()}_p_homophone_{p_homophone}_p_letter_{p_letter}_p_confusing_word_{p_confusing_word}.txt")

    def get_punctuation(self, word):
        #gets punctuationand symbols from a word
        punctuation = []
        for i in range(len(word)):
            if word[i] in '".,?!:;()' or word[i] == "'":
                punctuation.append((i,word[i]))
        return punctuation

    def homophone_swapper(self, in_word, out_word, apostrophe=False):
        #pick a random homophone from the list of homophones
        homophone = random.choice(self.homophones_dict[out_word])
        #check if difference is apostrophe
        if homophone.replace("'", "") == out_word:
            apostrophe = True
        #check if the first letter is capitalized
        if in_word.strip('".,?!:;()').strip("'")[0].isupper():
            #Capitalize the first letter of the homophone
            homophone = homophone[0].upper() + homophone[1:]
        #check if all the letters in the word are capitalized
        if in_word.isupper() and len(in_word) > 1:
            #capitalize the homophone
            homophone = homophone.upper()
        return homophone, apostrophe

    def confusing_word_injector(self, in_word, out_word):
        confusing_word = random.choice(self.confusing_words_dict[out_word])
        #check if the first letter is capitalized
        if in_word.strip('".,?!:;()').strip("'")[0].isupper():
            #Capitalize the first letter of the confusing word
            confusing_word = confusing_word[0].upper() + confusing_word[1:]
        #check if all the letters in the word are capitalized
        if in_word.isupper() and len(in_word) > 1:
            #capitalize the confusing word
            confusing_word = confusing_word.upper()
        return confusing_word

    def confusing_letter_swapper(self, in_word, out_word, p_letter, letters_swapped, homophone_swapped, confusing_word_swapped, confusing_letter_swapped):
        for i in range(len(out_word)):
                #check if swap a letter with a confusing letter with probability p_letter
                if  random.random() <= p_letter:
                    chance = random.random()
                    #need to skip first letter with 95% probability, based on Peddler findings
                    if i == 0 and chance >= 0.95:
                        continue
                    #check if the word is in the confusing letters dict
                    if out_word[i].lower() in self.confusing_letters_dict.keys():
                        #pick a random letter from the list of confusing letters
                        confusing_letter = random.choice(self.confusing_letters_dict[out_word[i].lower()])
                        #check if swapping a letter in a homophone
                        if not homophone_swapped and not confusing_word_swapped:
                            if in_word.strip('".,?!:;()').strip("'")[i].isupper():
                                confusing_letter = confusing_letter.upper()
                        else:
                            if out_word[i].isupper():
                                confusing_letter = confusing_letter.upper()
                        #replace the letter with the confusing letter at index j
                        out_word = out_word[:i] + confusing_letter + out_word[i+1:]
                        #update flag
                        confusing_letter_swapped = True
                        #update the amount of letters that were swapped
                        letters_swapped += 1
                        #lower the probability of swapping a letter with a confusing letter each time a letter is swapped
                        p_letter = 0.1*p_letter
                if not homophone_swapped and not confusing_word_swapped:
                    #ensure's proper capitalization of the word
                    if in_word.strip('".,?!:;()').strip("'")[i].isupper():
                        out_word = out_word[:i] + out_word[i].upper() + out_word[i+1:]
        return out_word, letters_swapped, confusing_letter_swapped


    def insert_punctuation(self, in_word, out_word, punctuation, apostrophe, homophone_swapped, confusing_word_swapped): #inwor car? outwor care
        if len(punctuation) == 0:
            return out_word
        for index, punc in punctuation:
                    if apostrophe and punc == "'" and index != 0 and index != len(in_word)-1:
                        continue
                    #if word already has that type of punctuation then skip it if its not at the first or last index
                    #also need to make sure its not punction that can be consecutive like "..." 
                    if index != 0 and index != len(in_word)-1 and punc in out_word:
                        if in_word[index-1] == punc and in_word[index+1] == punc:
                            out_word = out_word[:index] + punc + out_word[index:]
                        continue
                    if homophone_swapped or confusing_word_swapped:
                        if index == len(in_word)-1:
                            out_word = out_word + punc
                        continue
                    out_word = out_word[:index] + punc + out_word[index:]
        return out_word
    
    def injector(self, sentence, p_homophone, p_letter, p_confusing_word):
        #split the sentence into a list of words
        words = sentence.split()
        #keep track of the amount of homophones injected 
        homonphones_injected = 0
        #keep track of the amount of letters that were swapped
        letters_swapped = 0
        #keep track of the amount of confusing words injected
        confusing_words_injected = 0
        #keep track of the amount of words that were changed
        words_modified = 0
        for i in range(len(words)):
            #check for punctuation at all indexes of the word and save it
            punctuation = self.get_punctuation(words[i])         
            #get the word and remove any punctuation
            word = words[i].lower().strip('".,?!:;()')
            word = word.strip("'")
            #create a copy of the word to check if it was changed
            word_copy = words[i]
            #flag for homophone swapped and confusing letter swapped
            homophone_swapped = False
            confusing_letter_swapped = False
            confusing_word_swapped = False
            #flag for apostrophe
            apostrophe = False
            #check if the word is in the homophones dict
            if word in self.homophones_dict.keys():
                #check if the word has any homophones
                if len(self.homophones_dict[word]) > 0:
                    #swap the word with a homophone with probability p_homophone
                    if random.random() <= p_homophone:
                        #replace the word with the homophone, flag to see if apostrophe is the difference in homophone
                        word, apostrophe = self.homophone_swapper(words[i], word)
                        #update the amount of words that were swapped for homophone
                        homonphones_injected += 1
                        #update the flag
                        homophone_swapped = True
            #check if the word is in the pedler dict
            if word in self.confusing_words_dict.keys():
                #check if the word has any confusing words
                if len(self.confusing_words_dict[word]) > 0:
                    #swap the word with a homophone with probability p_homophone
                    if random.random() <= p_confusing_word:
                        word = self.confusing_word_injector(words[i], word)
                        #update the amount of words that were swapped for homophone
                        confusing_words_injected += 1
                        #update the flag
                        confusing_word_swapped = True
            #use confusing letter swapper to swap letters with probability p_letter
            word, letters_swapped, confusing_letter_swapped = self.confusing_letter_swapper(
                words[i], word, p_letter,
                letters_swapped, homophone_swapped,
                confusing_word_swapped, confusing_letter_swapped)
            #If whole word is upper case and its more than 1 letter then capitalize the whole word
            if words[i].isupper() and len(words[i]) > 1:
                word = word.upper()
            #replace the orignal word with new word and proper punctuation if any
            word = self.insert_punctuation(words[i], word, punctuation, apostrophe, homophone_swapped, confusing_word_swapped)
            if confusing_letter_swapped or homophone_swapped or confusing_word_swapped:
                words[i] = word
                words_modified += 1
        #join the list of words back into a sentence
        sentence = " ".join(words)
        return sentence, (homonphones_injected, letters_swapped, confusing_words_injected, words_modified)