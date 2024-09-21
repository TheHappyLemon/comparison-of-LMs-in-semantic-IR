from constants import PATH_LOGS, PATH_DATASET

lines = []
with open(PATH_LOGS + "search.log", 'r', encoding='utf-8') as f:
    lines = f.readlines()
with open(PATH_DATASET + "time_merged_pretified.log", 'w', encoding='utf-8') as out_file:
    for line in lines:
        params = line.split(' ')
        if params[7] == "Flushed" and params[9] == "search":
            continue
        out_file.write(line)