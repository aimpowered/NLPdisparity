import random
import DataLoader #Custom class
import pickle
import numpy as np

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
        df_swap_results = pd.DataFrame(columns=["dataset","p1", "p2", "words_swapped", "letters_swapped", "words_changed", "sentences_changed"])
        p1 = p_start
        p2 = p_start
        while p1 <= p_end:
            #create deep copy of the data
            if p1 == 0:
                temp_load, results = self.injection_runner(self.load.create_deepcopy(), p1, p2)
                df_swap_results.loc[len(df_swap_results)] = {"dataset":"wmt14_en", "p1":0, "p2":0, "words_swapped":results[0], "letters_swapped":results[1], "words_changed":results[2], "sentences_changed":results[3]}
                self.saver(temp_load, save_path, p1, p2, save_format)
            else:
                temp_load, results = self.injection_runner(self.load.create_deepcopy(), p1, p2)
                df_swap_results.loc[len(df_swap_results)] = {"dataset":"wmt14_en", "p1":p1, "p2":p2, "words_swapped":results[0], "letters_swapped":results[1], "words_changed":results[2], "sentences_changed":results[3]}
                self.saver(temp_load, save_path, p1, p2, save_format)

                temp_load, results = self.injection_runner(self.load.create_deepcopy(), p1, 0)
                df_swap_results.loc[len(df_swap_results)] = {"dataset":"wmt14_en", "p1":p1, "p2":0, "words_swapped":results[0], "letters_swapped":results[1], "words_changed":results[2], "sentences_changed":results[3]}
                self.saver(temp_load, save_path, p1, 0, save_format)

                temp_load, results = self.injection_runner(self.load.create_deepcopy(), 0, p2)
                df_swap_results.loc[len(df_swap_results)] = {"dataset":"wmt14_en", "p1":0, "p2":p2, "words_swapped":results[0], "letters_swapped":results[1], "words_changed":results[2], "sentences_changed":results[3]}
                self.saver(temp_load, save_path, 0, p2, save_format)

            #update the p1 and p2
            p1 += step_size
            p2 += step_size
        #save the results
        df_swap_results.to_csv(f"{save_path}/swap_results.csv", index=False)
        return df_swap_results


    def injection_runner(self, data_loader, p1, p2):
        #we need to keep track of the amount of words that were swapped
        words_swapped = 0
        #we also need to keep track of the amount of letters that were swapped
        letters_swapped = 0
        #we also need to keep track of the amount of words that were changed
        words_changed = 0
        #we need to keep track of the amount of sentences that were changed
        sentences_changed = 0

        for i in range(len(data_loader.data)):
            #we need to get the sentence
            sentence = data_loader.data[i]
            #we need to swap the sentence
            sentence, results = self.injector(sentence, p1, p2)
            #we need to update the amount of words that were swapped
            words_swapped += results[0]
            #we need to update the amount of letters that were swapped
            letters_swapped += results[1]
            #we need to update the amount of words that were changed
            words_changed += results[2]
            #we need to update the amount of sentences that were changed
            if results[2] > 0:
                sentences_changed += 1
            #we need to update the sentence in the dataframe
            data_loader.data[i] = sentence
        print(f"P1 = {p1}, P2 = {p2}")
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
    
    def saver(self, temp_load: DataLoader, save_path, p1, p2, format="both"):
        if format == "csv":
            temp_load.save_as_csv(save_path + f"{temp_load.get_name()}_p1_{p1}_p2_{p2}.csv")
        elif format == "txt":
            temp_load.save_as_txt(save_path + f"{temp_load.get_name()}_p1_{p1}_p2_{p2}.txt")
        else:
            #save as both
            temp_load.save_as_csv(save_path + f"{temp_load.get_name()}_p1_{p1}_p2_{p2}.csv")
            temp_load.save_as_txt(save_path + f"{temp_load.get_name()}_p1_{p1}_p2_{p2}.txt")

    def injector(self, sentence, p1, p2):
        #first we need to split the sentence into a list of words
        words = sentence.split()
        #we need to keep track of the amount of words that were swapped
        words_swapped = 0
        #we also need to keep track of the amount of letters that were swapped
        letters_swapped = 0
        #we also need to keep track of the amount of words that were changed
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
            #we need to check if the word is in the homophones dict
            if word in self.homophones_dict.keys():
                #we need to check if the word has any homophones
                if len(self.homophones_dict[word]) > 0:
                    #we need to check if we will swap the word with a homophone with probability p1
                    
                    if random.random() < p1:
                        print(p1)
                        #we need to pick a random homophone from the list of homophones
                        homophone = random.choice(self.homophones_dict[word])
                        #we need to swap the word with the homophone
                        words[i] = homophone
                        #we need to update the amount of words that were swapped
                        words_swapped += 1
                        continue
            p_letter = p2 #lower the probability of swapping a letter with a confusing letter each time we swap a letter
            for j in range(len(word)):
                #we need to check if we will swap a letter with a confusing letter
                if random.random() < p_letter:
                    #we need to check if the word is in the confusing letters dict
                    if word[j] in self.confusing_letters_dict.keys():
                        #we need to pick a random letter from the list of confusing letters
                        confusing_letter = random.choice(self.confusing_letters_dict[word[j]])
                        #replace the letter with the confusing letter at index j
                        word = word[:j] + confusing_letter + word[j+1:]
                        words[i] = word
                        #we need to update the amount of letters that were swapped
                        letters_swapped += 1
                        p_letter = 0.1*p_letter
            #we need to check if the word was changed
            if word != word_copy:
                words_changed += 1

        #we need to join the list of words back into a sentence
        sentence = " ".join(words)
        return sentence, (words_swapped, letters_swapped, words_changed)
        