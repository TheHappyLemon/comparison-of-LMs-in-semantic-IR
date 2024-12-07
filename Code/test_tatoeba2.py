import csv
from constants import PATH_MAIN

file_path = PATH_MAIN + "lvs_sentences.tsv"
total_words = 0
total_sentences = 0

with open(file_path, "r", encoding="utf-8") as file:
    reader = csv.reader(file, delimiter="\t")
    for row in reader:
        sentence = row[2]
        word_count = len(sentence.split())  # Split sentence by spaces to count words
        total_words += word_count
        total_sentences += 1

average_length = total_words / total_sentences

print(f"Total sentences: {total_sentences}")
print(f"Total words: {total_words}")
print(f"Average sentence length: {average_length:.2f} words")
