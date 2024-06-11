# NLPdisparity
Code and data for audit NLP models for performance disparity

### To access data
The data is available in the to_test folder. This folder includes the syntehtic dyslexia injected English text and the translated output from the services menmtioned in the paper.

Each file within the dataset consists of a “.txt” or “.docx” file containing the translated sentences from AWS, Google, Azure and OpenAI. Each line represents a translated sentence. The file names indicate the type of synthetic injection that was done to the English version and the associated injection probability. The “default” directory consists of the English versions that were submitted to the translation services. The “v1” and “v2” folder names can be ignored. File names and the folder name indicated the type and probability of injection. Each file is the same but with different varying levels/types of injections. E.g. the file name “wmt14_en_p_homophone_0.2_p_letter_0.0_p_confusing_word_0.0” has a probability of 20% to inject a homophone in a sentence, 0 % of injecting a confusing letter and 0% to inject a confusing word. The injection process is explained in our paper.

### Notable classes
* *Injecting_Dyslexia.ipynb* was used to inject synthetic dyslexi style text data
* *baseline_results.ipynb* is the notebook that looks over preliminary results (BLEU and WER)
* *edit_distance.ipynb* is the notebook where edit distance was calculated and includes some analysis
* *bert_score.ipynb* was used to calculate BERT score (bert_scores folder contains the saved scores for quicker access) and has some analysis
* *LaBSE_data.ipynb* was used to calculate LaBSE embeddings and has some analysis (LaBSE folder contains the saved results for quicker access)
* *pos_tags_analysis.ipynb* is an analysis of the POS tags of the translated text
* *diff_lib_analysis.ipynb* is the notebook that uses DiffLib to analyze the outputs from the translation services
* *swap_results_combined.csv* contains the results of the injection statistics for the injection text files that were translated and investigated
* *Datasheet for datasets* contains all important information regarding the dataset
* *investigating_translations.ipynb* is the notebook where the translations were analyzed at a sentence level
* All python files (*DataLoader, DyslexiaInjector and TestInjector*) can be used to inject dyslexia into text and test the translation services
* *to_test* folder contains all the data outputted from the translation services
* *data* folder contains the dictionaries used for the injection process (mentioned in the paper)