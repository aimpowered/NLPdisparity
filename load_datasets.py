from datasets import load_dataset
dataset_wmt_enfr = load_dataset("wmt14",'fr-en', split='test')
dataset_wmt_ende = load_dataset("wmt16",'de-en', split='test')

to_translate_wmt14_en = []
to_translate_wmt16_en = []
reference_wmt14_fr = []
reference_wmt16_de = []

for i in range(len(dataset_wmt_enfr)):
    to_translate_wmt14_en.append(dataset_wmt_enfr[i]['translation']['en'])
    reference_wmt14_fr.append(dataset_wmt_enfr[i]['translation']['fr'])
for i in range(len(dataset_wmt_ende)):
    to_translate_wmt16_en.append(dataset_wmt_ende[i]['translation']['en'])
    reference_wmt16_de.append(dataset_wmt_ende[i]['translation']['de'])


#dowload all the datasets in csv format