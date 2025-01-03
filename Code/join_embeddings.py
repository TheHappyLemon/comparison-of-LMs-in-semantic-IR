import h5py
import logging
from constants import PATH_DATASET, PATH_LOGS

# File used for merging data of multiple hdf5 files into a single one.

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    handlers=[
        logging.FileHandler(PATH_LOGS + "merge_LASER.log")
    ]
)

def copy_dataset(source_file_path, source_dataset_path, target_file_path, target_dataset_path):
    with h5py.File(source_file_path, 'r') as source_file:
        data = source_file[source_dataset_path]
        with h5py.File(target_file_path, 'a') as target_file:
            #if target_dataset_path in target_file:
                #del target_file[target_dataset_path]
                #print('deleted', target_dataset_path)
            target_file.create_dataset(target_dataset_path, data=data)
            logger.info(f"Data copied from file {source_file_path} dataset {source_dataset_path} to file {target_dataset_path} dataset {target_file_path}")

source_files = {
    '1' : '/LASER/introduction/en_cirrussearch',
    '2' : '/LASER/introduction/lv_cirrussearch',
    '3' : '/LASER/full-text/en_cirrussearch',
    '4' : '/LASER/full-text/lv_cirrussearch',
    '5' : '/LASER/title/en_cirrussearch',
    '6' : '/LASER/title/lv_cirrussearch'
}

target_file = PATH_DATASET + 'embeddings.hdf5'

for source_file in source_files:
    logger = logging.getLogger(__name__)
    copy_dataset(PATH_DATASET + 'embeddings_old.hdf5', source_files[source_file], target_file, source_files[source_file])
    