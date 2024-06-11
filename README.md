# NLPdisparity
Code and data for audit NLP models for performance disparity

### To access data
The data is available in the to_test folder. This folder includes the syntehtic dyslexia injected English text and the translated output from the services menmtioned in the paper.

Each file within the dataset consists of a “.txt” or “.docx” file containing the translated sentences from AWS, Google, Azure and OpenAI. Each line represents a translated sentence. The file names indicate the type of synthetic injection that was done to the English version and the associated injection probability. The “default” directory consists of the English versions that were submitted to the translation services. The “v1” and “v2” folder names can be ignored. File names and the folder name indicated the type and probability of injection. Each file is the same but with different varying levels/types of injections. E.g. the file name “wmt14_en_p_homophone_0.2_p_letter_0.0_p_confusing_word_0.0” has a probability of 20% to inject a homophone in a sentence, 0 % of injecting a confusing letter and 0% to inject a confusing word. The injection process is explained in our paper.
