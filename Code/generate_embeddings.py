import torch
import logging
import h5py
import csv
import numpy as np

from json import load as j_load
from constants import PATH_SETUP, PATH_LOGS, PATH_DATASET
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.FileHandler(PATH_LOGS + "main.log")#,
        #ogging.StreamHandler()
    ]
)

def get_dict_from_json(file : str):
    try:
        with open(file, 'r') as f:
            res = j_load(f)
            return res
    except FileNotFoundError:
        logger.error(f"File '{file}' not found!")
        return None

def get_wiki_pages_from_csv(file : str) -> dict:
    # Get info about each wiki page from dataset in format page_name : text_file
    result = {}

    with open(file, 'r', newline='\n', encoding='utf-8') as source:
        csv_reader = csv.DictReader(source, delimiter=';')
        header = csv_reader.fieldnames
        for line in csv_reader:
            result[line['en_title']] = line['file_name']
    return result

def create_hdf5_structure(file_name : str, models : dict, types : list, languages : list, wiki_pages : dict):

    models_dimensions = {}
    for model in models:
        model_instance = SentenceTransformer(models[model])
        embeddings_dimensions = model_instance.get_sentence_embedding_dimension()
        if embeddings_dimensions is None:
            logger.error(f"Method 'get_sentence_embedding_dimension' of model '{models[model]}' returns None!")
            return
        models_dimensions[model] = embeddings_dimensions
            
    with h5py.File(file_name, 'a') as file:
        for model in models:
            print(model)
            model_group = file.create_group(model)
            for type in types:
                type_group = model_group.create_group(type)
                for language in languages:
                    r = type_group.create_dataset(language, shape=(len(wiki_pages), models_dimensions[model]), chunks=(25, models_dimensions[model]))
                    logger.info(f"Created dataset '{r.name}'")
        file.create_dataset('mapping', shape=(len(wiki_pages), 1), chunks=(25, 1), data=list(wiki_pages.values()))
        logger.info(f"Created mapping dataset")
        logger.info(f"Finished creating hdf5 structure")

def fill_hdf5_structure(file_name : str, models : dict, types : list, languages : list):
    for model in models:
        for language in languages:
            for type in types:
                logger.info(f"Transforming '{language}_{type}' using '{model}'")

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    wiki_pages = get_wiki_pages_from_csv(PATH_DATASET + 'results_cirrussearch_ALL_common.csv')
    wiki_types = ['title', 'open', 'source']
    wiki_languages = ['en_cirrussearch', 'lv_cirrussearch']
    models = get_dict_from_json(PATH_SETUP + "models.json")

    #create_hdf5_structure(PATH_DATASET + "embeddings.hdf5", models, wiki_types, wiki_languages, wiki_pages)
    fill_hdf5_structure(PATH_DATASET + "embeddings.hdf5", models, wiki_types, wiki_languages)

    #for model in models:
        #language_model = SentenceTransformer(models[model])
        #logger.info(f"Instantiated model '{models[model]}'")
        #print(models[model], language_model.get_sentence_embedding_dimension())

    # multi process encoding
    #model = SentenceTransformer("all-MiniLM-L6-v2")
    #pool = model.start_multi_process_pool()
    #emb = model.encode_multi_process(sentences, pool, batch_size=25)
    #model.stop_multi_process_pool(pool)
