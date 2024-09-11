import h5py
from constants import PATH_DATASET, PATH_LOGS
import numpy as np

def get_dataset_names(hdf5_file : str) -> dict:
    datasets = []
    
    with h5py.File(hdf5_file, 'r') as file:
        for model in file.keys():
            if model == 'mapping':
                continue
            for type in file.get(model):
                source_dataset = f"/{model}/{type}/en_cirrussearch"
                query_dataset  = f"/{model}/{type}/lv_cirrussearch"
                datasets.append(source_dataset)
                datasets.append(query_dataset)
    return datasets

tolerance = 1e-6
i = 0
j = 0
hdf5_file = PATH_DATASET + "embeddings.hdf5"
datasets = get_dataset_names(hdf5_file)

if __name__ == "__main__":
    with open(PATH_LOGS + f"magnitudes_{tolerance}.csv", 'w', encoding='utf-8') as output_file:
        output_file.write(f"Dataset;magnitudes_not_one;indexes\n")
        with h5py.File(hdf5_file, 'r') as file:
            for dataset in datasets:
                i = i + 1
                j = 0
                index = 0
                bad_indexes = []
                for vector in file[dataset]:
                    length = np.linalg.norm(vector)
                    if abs(length - 1) > tolerance:
                        j = j + 1
                        bad_indexes.append(index)
                    index = index + 1
                output_file.write(f"{dataset};{j};{','.join(map(str, bad_indexes))}\n")
                print(f"{i})", dataset, j)