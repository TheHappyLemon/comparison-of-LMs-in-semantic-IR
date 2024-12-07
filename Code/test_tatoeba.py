from datasets import load_dataset

dataset = load_dataset("tatoeba", lang1="en", lang2="lv")

train_data = dataset["train"]

total_en = 0
total_lv = 0

for entry in train_data:
    en_sentence_length = len(entry["translation"]["en"].split())
    lv_sentence_length = len(entry["translation"]["lv"].split())
    total_en = total_en + en_sentence_length
    total_lv = total_lv + lv_sentence_length

avg_len_en = total_en / len(train_data)
avg_len_lv = total_lv / len(train_data)

print(f"Average English sentence length: {avg_len_en:.2f} words")
print(f"Average Latvian sentence length: {avg_len_lv:.2f} words")

# in MTEB paper they use abbreviation LVS-ENG, but here https://object.pouta.csc.fi/OPUS-Tatoeba I managed to only find en-lv dataset.
# They probably manually downloaded version from https://tatoeba.org/en/downloads