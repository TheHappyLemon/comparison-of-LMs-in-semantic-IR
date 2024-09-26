from constants import PATH_SETUP, PATH_LOGS, PATH_DATASET, BATCH_SIZE, ALLOWED_DIRS
import logging
import os
import csv
import statistics

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.FileHandler(PATH_LOGS + "calc_text_length_stats.log")
    ]
)

class TextType:
    
    def __init__(self) -> None:
        # Structure:
        # text_type : [len of 000000.txt, len of 000001.txt ... len of xxxxxx.txt]
        self.texts = {}

    def calculate_text_type_length(self, root : str, dir_name : str, files : list):
        
        if dir_name in self.texts:
            return

        logger.info(f"Calculating length of texts for '{dir_name}'")
        magnitudes = []
        self.texts[dir_name] = magnitudes

        for file in files:
            with open(root + os.path.sep + file, 'r', encoding='utf-8') as current_file:
                text = current_file.readlines()
                text = text[0]
                logger.info(f"{dir_name}: Length of {file} is {len(text)}")
                magnitudes.append(len(text))

    def map_namings(self, type : str) -> str:
        if type == "title":
            return "Title"
        elif type == "open":
            return "Introduction"
        elif type == "source":
            return "Body"

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    text_type = TextType()
    csv_header = ["Language", "Text_type", "Mean", "Median", "Standard_deviation", "texts_amount", "total_length", "min_length", "max_length"]

    with open(PATH_DATASET + "text_magnitudes.csv", 'w', encoding='utf-8', newline='') as csv_file:

        csv_writer = csv.DictWriter(csv_file, delimiter=';', fieldnames=csv_header)
        csv_writer.writeheader()
        csv_rows = []


        for root, dirs, files in os.walk(PATH_DATASET):
            dir_name = root.split("\\")[-1]
            if dir_name in ALLOWED_DIRS:

                csv_row = {}
                params  = dir_name.split("_")
                csv_row["Language"]  = params[0]
                csv_row["Text_type"] = text_type.map_namings(params[2])

                text_type.calculate_text_type_length(root=root, dir_name=dir_name, files=files)
                
                csv_row["texts_amount"] = len(text_type.texts[dir_name])
                csv_row["total_length"] = sum(text_type.texts[dir_name])
                csv_row["min_length"]   = min(text_type.texts[dir_name])
                csv_row["max_length"]   = max(text_type.texts[dir_name])

                csv_row["Mean"]               = statistics.mean(text_type.texts[dir_name])
                csv_row["Median"]             = statistics.median_grouped(text_type.texts[dir_name])
                csv_row["Standard_deviation"] = statistics.stdev(text_type.texts[dir_name])

                logger.info(f"Info to csv -> {str(csv_row)}")
                csv_rows.append(csv_row)
        csv_writer.writerows(csv_rows)