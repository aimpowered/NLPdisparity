import random
import pickle
import numpy as np
import DataLoader #Custom class

class DyslexiaInjector:
    def __init__(self, load: DataLoader, homophone_path = "", confusing_letters_path = "", seed = 42):
        self.load = load
        self.homophones_dict = self.load_homophones(homophone_path)
        self.confusing_letters_dict = self.load_confusing_letters(confusing_letters_path)
        random.seed(seed)

    def load_homophones(self, path):
        with open(path, "rb") as f:
            homophone_dict = pickle.load(f)
            #close the file
            f.close()
        return homophone_dict
    
    def load_confusing_letters(self, path):
        with open(path, "rb") as f:
            confusing_letters_dict = pickle.load(f)
            #close the file
            f.close()
        return confusing_letters_dict
    
    def injection_swap(self, p_start=0, p_end=1, step_size=0.1, save_path="", save_format="both"):
        df_swap_results = pd.DataFrame(columns=["dataset","p_homophone", "p_letter", "words_swapped",
                                        "letters_swapped", "words_changed", "sentences_changed"])
        p_homophone = p_start
        p_letter = p_start
        while p_homophone <= p_end:
            #create deep copy of the data
            if p_homophone == 0:
                temp_load, results = self.injection_runner(self.load.create_deepcopy(), p_homophone, p_letter)
                df_swap_results.loc[len(df_swap_results)] = {"dataset":"wmt14_en", "p_homophone":0, "p_letter":0, "words_swapped":results[0],
                                                             "letters_swapped":results[1], "words_changed":results[2], "sentences_changed":results[3]}
                self.saver(temp_load, save_path, p_homophone, p_letter, save_format)
            else:
                temp_load, results = self.injection_runner(self.load.create_deepcopy(), p_homophone, p_letter)
                df_swap_results.loc[len(df_swap_results)] = {"dataset":"wmt14_en", "p_homophone":p_homophone, "p_letter":p_letter, "words_swapped":results[0],
                                                             "letters_swapped":results[1], "words_changed":results[2], "sentences_changed":results[3]}
                self.saver(temp_load, save_path, p_homophone, p_letter, save_format)

                temp_load, results = self.injection_runner(self.load.create_deepcopy(), p_homophone, 0)
                df_swap_results.loc[len(df_swap_results)] = {"dataset":"wmt14_en", "p_homophone":p_homophone, "p_letter":0, "words_swapped":results[0],
                                                             "letters_swapped":results[1], "words_changed":results[2], "sentences_changed":results[3]}
                self.saver(temp_load, save_path, p_homophone, 0, save_format)

                temp_load, results = self.injection_runner(self.load.create_deepcopy(), 0, p_letter)
                df_swap_results.loc[len(df_swap_results)] = {"dataset":"wmt14_en", "p_homophone":0, "p_letter":p_letter, "words_swapped":results[0],
                                                             "letters_swapped":results[1], "words_changed":results[2], "sentences_changed":results[3]}
                self.saver(temp_load, save_path, 0, p_letter, save_format)

            #update the p_homophone and p_letter
            p_homophone += step_size
            p_letter += step_size
        #save the results
        df_swap_results.to_csv(f"{save_path}/swap_results.csv", index=False)
        return df_swap_results


    def injection_runner(self, data_loader, p_homophone, p_letter):
        #track of the amount of words that were swapped
        words_swapped = 0
        #keep track of the amount of letters that were swapped
        letters_swapped = 0
        #track of the amount of words that were changed
        words_changed = 0
        #track of the amount of sentences that were changed
        sentences_changed = 0

        for i in range(len(data_loader.data)):
            #get the sentence
            sentence = data_loader.data[i]
            #swap the sentence
            sentence, results = self.injector(sentence, p_homophone, p_letter)
            # update the amount of words that were swapped
            words_swapped += results[0]
            #update the amount of letters that were swapped
            letters_swapped += results[1]
            #update the amount of words that were changed
            words_changed += results[2]
            #update the amount of sentences that were changed
            if results[2] > 0:
                sentences_changed += 1
            #update the sentence in the dataframe
            data_loader.data[i] = sentence
        print(f"p_homophone = {p_homophone}, p_letter = {p_letter}")
        print("Words swapped: " + str(words_swapped))
        print("Letters swapped: " + str(letters_swapped))
        print("Words changed: " + str(words_changed))
        print("Sentences changed: " + str(sentences_changed))
        return data_loader, (words_swapped, letters_swapped, words_changed, sentences_changed)

    def get_homophones(self, word):
        return self.homophones_dict[word]
    
    def get_homophone_dict(self):
        return self.homophones_dict
    
    def get_confusing_letters(self, word):
        return self.confusing_letters_dict[word]
    
    def get_confusing_letters_dict(self):
        return self.confusing_letters_dict
    
    def saver(self, temp_load: DataLoader, save_path, p_homophone, p_letter, format="both"):
        if format == "csv":
            temp_load.save_as_csv(save_path + f"{temp_load.get_name()}_p_homophone_{p_homophone}_p_letter_{p_letter}.csv")
        elif format == "txt":
            temp_load.save_as_txt(save_path + f"{temp_load.get_name()}_p_homophone_{p_homophone}_p_letter_{p_letter}.txt")
        else:
            #save as both
            temp_load.save_as_csv(save_path + f"{temp_load.get_name()}_p_homophone_{p_homophone}_p_letter_{p_letter}.csv")
            temp_load.save_as_txt(save_path + f"{temp_load.get_name()}_p_homophone_{p_homophone}_p_letter_{p_letter}.txt")

    def injector(self, sentence, p_homophone, p_letter):
        #split the sentence into a list of words
        words = sentence.split()
        #keep track of the amount of words that were swapped
        words_swapped = 0
        #keep track of the amount of letters that were swapped
        letters_swapped = 0
        #keep track of the amount of words that were changed
        words_changed = 0
        #debugging purposes, alows us to see which words were modified and how
        #changed_words = []
        for i in range(len(words)):
            #check for punctuation at all indexes of the word and save it, so it can be added back later
            punctation = []
            for j in range(len(words[i])):
                if words[i][j] in '".,?!:;()' or words[i][j] == "'":
                    punctation.append((j,words[i][j]))
            #get the word and remove any punctuation
            word = words[i].lower().strip('".,?!:;()')
            word = word.strip("'")
            #create a copy of the word so we can check if it was changed
            word_copy = words[i]
            #flag for homophone swapped and confusing letter swapped
            homophone_swapped = False
            confusing_letter_swapped = False
            #check if the word is in the homophones dict
            if word in self.homophones_dict.keys():
                #check if the word has any homophones
                if len(self.homophones_dict[word]) > 0:
                    #check if we will swap the word with a homophone with probability p_homophone
                    if random.random() < p_homophone:
                        #pick a random homophone from the list of homophones
                        homophone = random.choice(self.homophones_dict[word])
                        #check if the first letter is capitalized then we need to capitalize the first letter of the homophone
                        #also gets the index of the first letter of the word 
                        if words[i].strip('".,?!:;()').strip("'")[0].isupper():
                            homophone = homophone[0].upper() + homophone[1:]
                        #check if all the letters in the word are capitalized
                        if words[i].isupper() and len(words[i]) > 1:
                            #capitalize the homophone
                            homophone = homophone.upper()
                        #replace the word with the homophone
                        word = homophone
                        #update the amount of words that were swapped for homophone
                        words_swapped += 1
                        #update the flag
                        homophone_swapped = True
                        continue
            p_letter_2 = p_letter
            for j in range(len(word)):
                #check if we will swap a letter with a confusing letter
                if random.random() < p_letter_2:
                    #check if the word is in the confusing letters dict
                    if word[j].lower() in self.confusing_letters_dict.keys():
                        #pick a random letter from the list of confusing letters
                        confusing_letter = random.choice(self.confusing_letters_dict[word[j].lower()])
                        #check if we are swapping in a homophone
                        if not homophone_swapped:
                            if words[i].strip('".,?!:;()').strip("'")[j].isupper():
                                confusing_letter = confusing_letter.upper()
                        else:
                            if word[j].isupper():
                                confusing_letter = confusing_letter.upper()
                        #replace the letter with the confusing letter at index j
                        word = word[:j] + confusing_letter + word[j+1:]
                        #update flag
                        confusing_letter_swapped = True
                        #update the amount of letters that were swapped
                        letters_swapped += 1
                        #lower the probability of swapping a letter with a confusing letter each time we swap a letter
                        p_letter_2 = 0.1*p_letter_2
                #ensure proper capitalization of the word
                if words[i].strip('".,?!:;()').strip("'")[j].isupper() and not homophone_swapped:
                    word = word[:j] + word[j].upper() + word[j+1:]
            #if entire word is upper case and its more than 1 letter then capitalize the whole word
            if words[i].isupper() and len(words[i]) > 1:
                word = word.upper()
            #replace the orignal word with new word and proper punctuation if any
            if len(punctation) > 0:
                for index, punc in punctation:
                    #if word already has that type of punctuation then skip it if its not at the first or last index
                    #also need to make sure its not punction that can be consecutive like "..." 
                    if index != 0 and index != len(words[i])-1 and punc in word:
                        if words[i][index-1] == punc and words[i][index+1] == punc:
                            word = word[:index] + punc + word[index:]
                        continue
                    word = word[:index] + punc + word[index:]
            if confusing_letter_swapped or homophone_swapped:
                words[i] = word
                #check if the word was changed
                if words[i] != word_copy:
                    words_changed += 1
                    #For debugging purpouses
                    #changed_words.append((word_copy, word))
        #join the list of words back into a sentence
        sentence = " ".join(words)
        #For debugging purpouses
        #print(changed_words)
        return sentence, (words_swapped, letters_swapped, words_changed)
