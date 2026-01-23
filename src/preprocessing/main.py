from .dataset_preprocessing import preprocess_ade20k
from src.utils.params import Params

global_params = Params("src/config/global_params.json")
preprocess_ade20k(global_params)