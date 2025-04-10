from end2you.data_provider import get_dataloader
from end2you.utils import Params
import numpy as np

provider_params = Params({
    'batch_size':4,
    'is_training':True,
    'cuda':True,
    'num_workers':2,
    'seq_length':150,
    'modality':'audio',
    'dataset_path':'MELD/train_hdf5s'
})

audio_provider = get_dataloader(provider_params)

for batch in audio_provider:
    print(batch)
    break
