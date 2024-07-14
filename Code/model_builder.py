from json import load as j_load
from logging import Logger
from sentence_transformers import SentenceTransformer

class ModelBuilder:
    def __init__(self, config_file : str) -> None:
        self.__models_dict = {}
        self.__models_instances = {}
        self.__config_file = config_file
        self.__parse_config__()
        self.__instantiate_models__()
    

    def __parse_config__(self) -> None:
        with open(self.__config_file, 'r') as f:
            self.__models_dict = j_load(f)
            return True
        # May cause FileNotFoundError, but I want it.
        
    def __instantiate_models__(self) -> None:
        for model in self.__models_dict:
            self.__models_instances[model] = SentenceTransformer(self.__models_dict[model])

    def get_models(self) -> dict:
        return self.__models_instances
    
    def get_model(self, model_name : str) -> SentenceTransformer:
        return self.__models_instances[model_name]
        # may cause KeyError, but I want it


    def __str__(self) -> str:
        pass