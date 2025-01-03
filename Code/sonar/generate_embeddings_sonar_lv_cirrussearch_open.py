import torch
import logging
import h5py
import csv
import numpy as np
import os
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

from constants_HPC import PATH_LOGS, PATH_DATASET, BATCH_SIZE

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.FileHandler(PATH_LOGS + "main_sonar_lv_open.log")
    ]
)

text2vec = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder",
                                        tokenizer="text_sonar_basic_encoder")

def get_np_array_zero_rows(np_array):
    return np.where(~np.any(np_array, axis=1))[0]

def get_wiki_pages_from_csv(file : str) -> dict:
    # Get info about each wiki page from dataset in format page_name : text_file
    result = {}

    with open(file, 'r', newline='\n', encoding='utf-8') as source:
        csv_reader = csv.DictReader(source, delimiter=';')
        header = csv_reader.fieldnames
        for line in csv_reader:
            result[line['en_title']] = line['file_name']
    return result

def get_files_data(directory_path, file_names):
    file_contents = []
    
    for file_name in file_names:
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                file_contents.append(file.read())
        else:
            file_contents.append(None)
            logger.error(f"File '{file_path}' not found! Appending 'None!'")
            
    return file_contents

def create_hdf5_structure(file_name : str, model : str, types : list, languages : list, wiki_pages : dict):

    model_dimension = 1024
            
    with h5py.File(file_name, 'a') as file:
            
        if model in file:
            return

        model_group = file.create_group(model)
        for type in types:
            type_group = model_group.create_group(type)
            for language in languages:
                r = type_group.create_dataset(language, shape=(len(wiki_pages), model_dimension), chunks=(25, model_dimension))
                logger.info(f"Created dataset '{r.name}' with shape '{r.shape[0]}x{r.shape[1]}'")

        if not 'mapping' in file:
            file.create_dataset('mapping', shape=(len(wiki_pages), 1), chunks=(25, 1), data=list(wiki_pages.values()))
            logger.info(f"Created mapping dataset")

def fill_hdf5_structure(file_name : str, model : str, types : list, languages : list):

    wiki_pages_names = []
    with h5py.File(file_name, 'r') as file:
        wiki_pages_names = file['mapping'][:]
        wiki_pages_names = [fn[0].decode() for fn in wiki_pages_names] # because they are in binary format

    for language in languages:
        src_lang = "eng_Latn" if language == "en_cirrussearch" else "lvs_Latn"
        for type in types:
            sub_folder = language + "_" + type

            with h5py.File(file_name, 'a') as file:
                dataset = file[model][type][language]

                offset = get_np_array_zero_rows(dataset)
                if len(offset) == 0:
                    logger.info(f"Dataset '{dataset.name}' is alredy finished!")
                    continue
                offset = offset[0]
                
                if sub_folder != "lv_cirrussearch_open":
                    continue
                
                logger.info(f"Transforming '{sub_folder}' using '{model}'")
                logger.info(f"Starting from row '{offset}'")

                # Get each text from a dataset in a batch of <BATCH_SIZE>, encode to embeddings and write to hdf5 file
                for i in range(offset, len(wiki_pages_names), BATCH_SIZE):
                    wiki_pages_names_batch = wiki_pages_names[i:i + BATCH_SIZE]
                    wiki_pages_data = get_files_data(os.path.join(PATH_DATASET, sub_folder), wiki_pages_names_batch)
                    
                    embeddings = text2vec.predict(wiki_pages_data, source_lang = src_lang) 
                    dataset[i: i + BATCH_SIZE] = embeddings 
                    
                    #if i > 100:
                    #    return

                    if i % 1000 == 0:
                        logger.info(f"Processed a thousand pages!")
                    i = i + BATCH_SIZE


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    wiki_pages = get_wiki_pages_from_csv(PATH_DATASET + 'results_cirrussearch_ALL_common.csv')
    wiki_types = ['title', 'introduction', 'full-text']
    wiki_languages = ['en_cirrussearch', 'lv_cirrussearch']

    create_hdf5_structure(PATH_DATASET + "embeddings_sonar_lv_open.hdf5", "SONAR", wiki_types, wiki_languages, wiki_pages)
    fill_hdf5_structure(PATH_DATASET + "embeddings_sonar_lv_open.hdf5", "SONAR", wiki_types, wiki_languages)