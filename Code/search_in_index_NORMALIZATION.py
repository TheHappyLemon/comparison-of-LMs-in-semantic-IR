import h5py
import logging
from constants import PATH_LOGS, PATH_DATASET, CHUNK_SIZE, PATH_SEARCH_NORMALIZED_HNSW, PATH_SEARCH_NORMALIZED_IVF, PATH_SEARCH_NORMALIZED_FLATL2,  PATH_SETUP
import faiss
import numpy as np
import csv
import os
from io import TextIOWrapper
from datetime import datetime
from json import load as j_load

csv_header    = ['query_file', 'found', 'search_result', 'search_distances']
ignore_zeroes = True
index_types_allowed = ['FlatL2', 'HNSWFlat', 'IVFFlat'] 
index_types   = [] 

def get_np_array_zero_rows(np_array):
    return np.where(~np.any(np_array, axis=1))[0]

def build_index_from_hdf(file_path : str, dataset : str, index_type : str) -> faiss.IndexHNSWFlat:
    with h5py.File(file_path, 'r') as file:

        if not dataset in file:
            logger.error(f"Dataset {dataset} not in file")
            return
        # np.any - non zero rows, ~ - invert, np.where - return array of indexes with True values
        zero_rows_indexes = get_np_array_zero_rows(file[dataset])
        if len(zero_rows_indexes) != 0:
            logger.error(f"Embeddings in dataset '{dataset}' on index '{zero_rows_indexes[0]}' is not calculated!!!")
            exit()
        limit = file[dataset].shape[0]
        
        dimensions = file[dataset].shape[1]
        normalized_vectors = file[dataset][:limit]
        faiss.normalize_L2(normalized_vectors)

        if index_type == 'FlatL2':
            index = faiss.IndexFlatL2(dimensions)
            index.add(normalized_vectors)
        if index_type == 'FlatIP':
            index = faiss.IndexFlatIP(dimensions)
            index.add(normalized_vectors)
        if index_type == 'HNSWFlat':
            index = faiss.IndexHNSWFlat(dimensions, 64)
            index.add(normalized_vectors)
        if index_type == 'IVFFlat':
            quantizer = faiss.IndexFlatIP(dimensions)
            index = faiss.IndexIVFFlat(quantizer, dimensions, 128)
            index.train(normalized_vectors)
            index.add(normalized_vectors)
            index.nprobe = 8
        return index

def get_dataset_names(hdf5_file : str, only_model : str = "") -> dict:
    datasets = {}
    
    with h5py.File(hdf5_file, 'r') as file:
        for model in file.keys():
            if model == 'mapping':
                continue

            if only_model != "" and model != only_model:
                continue    

            for type in file.get(model):
                source_dataset = f"/{model}/{type}/en_cirrussearch"
                query_dataset  = f"/{model}/{type}/lv_cirrussearch"
                datasets[source_dataset] = query_dataset
    return datasets

def search_in_dataset(path_to_csv : str, index : faiss.IndexHNSWFlat, query_dataset : str, k : int, file : h5py.File) -> None:

    csv_rows = []

    with open(path_to_csv, 'w', newline='\n', encoding='utf-8') as result_csv:
        csv_writer = csv.DictWriter(result_csv, delimiter=';', fieldnames=csv_header)
        csv_writer.writeheader()

        for i in range(len(file[query_dataset])):
            
            csv_row = {}
            query_file  = file['mapping'][i][0].decode()

            # https://github.com/facebookresearch/faiss/issues/493
            normalized_vector = np.array([file[query_dataset][i]])
            faiss.normalize_L2(normalized_vector)
            Distances, Indexes = index.search(normalized_vector, k=k)
            Distances = Distances[0]
            Indexes = Indexes[0]

            search_result = [file['mapping'][Indexes[j]][0].decode() for j in range(len(Indexes))]
            search_distances = [str(Distances[j]) for j in range(len(Distances))]
            search_result = ','.join(search_result)
            search_distances = ','.join(search_distances)

            csv_row['search_result'] = search_result
            csv_row['search_distances'] = search_distances
            csv_row['found'] = (query_file in search_result)
            csv_row['query_file'] = query_file

            csv_rows.append(csv_row)
            if i % CHUNK_SIZE == 0 and i != 0:
                csv_writer.writerows(csv_rows)
                csv_rows.clear()
                logger.info(f"Flushed '{CHUNK_SIZE}' search results to csv file")
        if len(csv_rows) != 0:
            csv_writer.writerows(csv_rows)
            logger.info(f"Flushed '{len(csv_rows)}' search results to csv file")

if __name__ == '__main__':

    with open(PATH_SETUP + "models.json", 'r') as f:
        models_dict = j_load(f)
    models_dict["LASER"] = "null"
    models_dict["SONAR"] = "null"
    choosen_model = input("Enter model name: ")
    if choosen_model == "ALL":
        choosen_model = ""
    elif choosen_model not in models_dict:
        print(f"Bad model choosen!\nOnly: '{', '.join(sorted(list(models_dict.keys())))}'")
        exit()

    choosen_index = input("Enter index name: ")
    if choosen_index not in index_types_allowed:
        print(f"Bad index choosen!\nOnly: '{', '.join(sorted(index_types_allowed))}'")
        exit()
    index_types.append(choosen_index)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
        handlers=[
            logging.FileHandler(PATH_LOGS + f"search_NORMALIZED_{choosen_model}_{choosen_index}.log")
        ]
    )

    logger    = logging.getLogger(__name__)
    hdf5_file = PATH_DATASET + "embeddings.hdf5"
    kNN       = [1, 5, 10, 20]

    datasets  = get_dataset_names(hdf5_file, choosen_model)

    for index_type in index_types:
        
        if index_type == 'FlatL2':
            path_search_result = PATH_SEARCH_NORMALIZED_FLATL2
        elif index_type == 'HNSWFlat':
            path_search_result = PATH_SEARCH_NORMALIZED_HNSW
        elif index_type == 'IVFFlat':
            path_search_result = PATH_SEARCH_NORMALIZED_IVF
        else:
            logger.error(f"Unmapped index type f{index_type}!")
            exit()

        logger.info(f"Start working on index type {index_type}")

        for dataset in datasets:
            index = build_index_from_hdf(hdf5_file, dataset, index_type)
            logger.info(f"Succefully created index for dataset '{dataset}'")
            with h5py.File(hdf5_file, 'r') as file:
                query_dataset = datasets[dataset]
                for k in kNN:
                    path_to_dir = os.path.join(path_search_result, dataset.split('/')[1], dataset.split('/')[2])
                    os.makedirs(path_to_dir, exist_ok=True)
                    path_to_csv = os.path.join(path_to_dir, f"{k}NN.csv")
                    #path_to_csv = os.path.join(path_search_FLAT, dataset.split('/')[1], dataset.split('/')[2], f"{k}NN.csv")
                    start = datetime.now()
                    logger.info(f"Searching consequently for each element of '{query_dataset}'. k = '{k}'. Output to '{path_to_csv}'. Start time = '{start}'")
                    search_in_dataset(path_to_csv, index, query_dataset, k, file) 
                    logger.info(f"Done. Execution time = '{datetime.now() - start}'")