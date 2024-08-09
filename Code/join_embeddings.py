import h5py
from constants import PATH_DATASET

# File used for merging data of multiple hdf5 files into a single one.

def copy_dataset(source_file_path, source_dataset_path, target_file_path, target_dataset_path):
    with h5py.File(source_file_path, 'r') as source_file:
        if source_dataset_path in source_file:
            with h5py.File(target_file_path, 'a') as target_file:
                data = source_file[source_dataset_path]
                target_file.create_dataset(target_dataset_path, data=data)
                print(f"Data copied from {source_dataset_path} to {target_dataset_path} in {target_file_path}")
        else:
            print(f"Dataset {source_dataset_path} not found in {source_file_path}")

source_files = {
    'embeddings_sonar_en_open.hdf5'  : '/SONAR/open/en_cirrussearch',
    'embeddings_sonar_en_source.hdf5': '/SONAR/open/lv_cirrussearch',
    'embeddings_sonar_en_title.hdf5' : '/SONAR/title/en_cirrussearch',
    'embeddings_sonar_lv_open.hdf5'  : '/SONAR/title/lv_cirrussearch',
    'embeddings_sonar_lv_source.hdf5': '/SONAR/source/en_cirrussearch',
    'embeddings_sonar_lv_title.hdf5' : '/SONAR/source/lv_cirrussearch'
}
target_file = PATH_DATASET + 'embeddings.hdf5'

for source_file in source_files:
    copy_dataset(PATH_DATASET + source_file, source_files[source_file], target_file, source_files[source_file])
    print(source_file, '->', 'done')
