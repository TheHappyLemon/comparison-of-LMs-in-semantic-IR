import os
import csv
import logging
from constants import PATH_SEARCH_FLATIP, PATH_SEARCH_FLATL2, PATH_SEARCH_HNSWFLAT, PATH_SEARCH_IVFFLAT, PATH_SEARCH_FLATIP_NORMALIZED, PATH_SEARCH_NORMALIZED_HNSW, PATH_SEARCH_NORMALIZED_IVF, PATH_LOGS

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.FileHandler(PATH_LOGS + "statistics_HNSW_IVF_normalized.log")
    ]
)

def count_true_in_file(filepath):
    count = 0
    with open(filepath, 'r') as file:
        for line in file:
            count += line.count('True')
    return count


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    TOTAL = 77736
    index_types   = {
        'HNSWFlatNormalized' : PATH_SEARCH_NORMALIZED_HNSW,
        'IVFFlatNormalized'  : PATH_SEARCH_NORMALIZED_IVF
        #'FlatL2'           : PATH_SEARCH_FLATL2,
        #'FlatIP'           : PATH_SEARCH_FLATIP, 
        #'FlatIPNormalized' : PATH_SEARCH_FLATIP_NORMALIZED, 
        #'HNSWFlat'         : PATH_SEARCH_HNSWFLAT,
        #'IVFFlat'          : PATH_SEARCH_IVFFLAT
    }

    for index in index_types:
        result_file = f"Results_{index}_new.csv"
        path_to_dir = index_types[index]
        output_file = os.path.join(path_to_dir, result_file)
        i = 0
        logger.info(f"Start analyzing results for index type {index}")
        
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(["model", "type", "kNN", "successful_finds", "recall", "recall_percentage", "recall_percentage_rounded"])
            
            for root, dirs, files in os.walk(path_to_dir):
                #only csv, not output file, then sort by name length, if length same, then alhpabetically.
                files = sorted([f for f in files if f.endswith('.csv') and f != result_file and not f.endswith('_old.csv') ], key=lambda x: (len(x), x))
                if len(files) != 0:
                    logger.info(f"Analyzing {root}")

                for file in files:
                    filepath = os.path.join(root, file)
                    count = count_true_in_file(filepath)
                    
                    relative_path = filepath.split(os.path.sep)[-3::]
                    model = relative_path[0]
                    type  = relative_path[1]
                    kNN   = relative_path[2].split('.')[0]

                    recall = count / TOTAL
                    recall_percentage = recall * 100
                    recall_percentage_rounded = round(recall_percentage, 2)

                    logger.info(f"Results for {filepath}: '{model}', '{type}', '{kNN}', '{count}', '{recall}', '{recall_percentage}', '{recall_percentage_rounded}'")

                    writer.writerow([model, type, kNN, count, recall, recall_percentage, recall_percentage_rounded])
                