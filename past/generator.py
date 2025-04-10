from end2you.generation_process import GenerationProcess
from end2you.utils import Params

generator_params = Params(dict_params={
    'save_data_folder': 'MELD/train_hdf5s',
    'modality': 'audiovisual',
    'input_file': 'MELD/train_input.csv',
    'exclude_cols':'0',
    'delimiter': ',',
    'fieldnames':None,
    'log_file': 'generation.log',
    'root_dir': 'output_meld'
})
generation = GenerationProcess(generator_params)
generation.start()

