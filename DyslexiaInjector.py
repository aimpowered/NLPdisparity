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
        df_swap_results = pd.DataFrame(columns=["dataset","p_homophone", "p_letter", "words_swapped", "letters_swapped", "words_changed", "sentences_changed"])
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

    def injector(self, sentence, p_homophone, p_letter_2):
        #split the sentence into a list of words
        words = sentence.split()
        #keep track of the amount of words that were swapped
        words_swapped = 0
        #keep track of the amount of letters that were swapped
        letters_swapped = 0
        #keep track of the amount of words that were changed
        words_changed = 0
        for i in range(len(words)):
            #if the word has any punctuation or capitalization, we skip it
            if words[i].isalpha() == False:
                continue
            if words[i].islower() == False:
                continue
            word = words[i].lower().strip('".,?!:;()')
            #create a copy of the word so we can check if it was changed
            word_copy = word
            #check if the word is in the homophones dict
            if word in self.homophones_dict.keys():
                #check if the word has any homophones
                if len(self.homophones_dict[word]) > 0:
                    #check if we will swap the word with a homophone with probability p_homophone
                    if random.random() < p_homophone:
                        print(p_homophone)
                        #pick a random homophone from the list of homophones
                        homophone = random.choice(self.homophones_dict[word])
                        #swap the word with the homophone
                        words[i] = homophone
                        # update the amount of words that were swapped
                        words_swapped += 1
                        continue
            p_letter_2 = p_letter_2
            for j in range(len(word)):
                #check if we will swap a letter with a confusing letter
                if random.random() < p_letter_2:
                    #check if the word is in the confusing letters dict
                    if word[j] in self.confusing_letters_dict.keys():
                        #we need to pick a random letter from the list of confusing letters
                        confusing_letter = random.choice(self.confusing_letters_dict[word[j]])
                        #replace the letter with the confusing letter at index j
                        word = word[:j] + confusing_letter + word[j+1:]
                        #replace the orignal word with new word
                        words[i] = word
                        #update the amount of letters that were swapped
                        letters_swapped += 1
                        #lower the probability of swapping a letter with a confusing letter each time we swap a letter
                        p_letter_2 = 0.1*p_letter_2
            #check if the word was changed
            if word != word_copy:
                words_changed += 1

        #join the list of words back into a sentence
        sentence = " ".join(words)
        return sentence, (words_swapped, letters_swapped, words_changed)
        
