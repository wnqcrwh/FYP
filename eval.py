from config import Config
from FYP.data.meld_data import MELD_Dataset
from torch.utils.data import DataLoader
from modules.multi_modal_model import MultiModalModel
from utils import evaluate


C=Config()
test_dataset = MELD_Dataset(
    csv_path=C.test_csv_path,
    video_dir=C.test_video_dir,
    image_size=C.image_size,
    frame_rate=C.frame_rate,
    sr=C.sr,
    image_augment=C.image_augment,
    audio_augment=C.audio_augment,
    feature_type=C.feature_type,
    mode='test'
)
test_loader = DataLoader(
    test_dataset,
    batch_size=C.batch_size,
    shuffle=False,
    num_workers=C.num_workers
)
