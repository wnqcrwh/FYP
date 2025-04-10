import numpy as np
from pathlib import Path
import os
from natsort import natsorted

data_path=Path("MELD/train")
label_path=Path("MELD/train_csvs")


data_files = natsorted([f for f in os.listdir(data_path)])
label_files = natsorted([f for f in os.listdir(label_path)])

data_files_cleaned = [f.split('.')[0] for f in data_files]  # 去掉扩展名
label_files_cleaned = [f.split('.')[0] for f in label_files]

common_files = set(data_files_cleaned).intersection(set(label_files_cleaned))

data_files_aligned = [str(data_path / f) for f in data_files if f.split('.')[0] in common_files]
label_files_aligned = [str(label_path / f) for f in label_files if f.split('.')[0] in common_files]

files_array = np.column_stack([data_files_aligned, label_files_aligned])

path2save_file = Path('MELD/train_input.csv')
np.savetxt(str(path2save_file), files_array, delimiter=',', fmt='%s', header='raw_file,label_file')
