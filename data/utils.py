from config import Config
import torch

C=Config()
def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences in a batch.
    """
    batch = [b for b in batch if b is not None]
    

    # Unzip the batch into separate lists
    sequences, labels = zip(*batch)
    original_frames, video_frames, audio_frames = zip(*sequences)
    B = len(video_frames)
    # Pad sequences to the maximum length in the batch
    max_length_frame = max(f.shape[0] for f in video_frames)
    max_length_audio = max(f.shape[0] for f in audio_frames)

    padded_video = torch.zeros((B, max_length_frame, 3, *C.image_size), dtype=torch.float32)
    padded_origin = torch.zeros((B, max_length_frame, *original_frames[0].shape[1:]), dtype=torch.float32)
    padded_audio = torch.zeros((B, max_length_audio, 1, 128, 128), dtype=torch.float32)

    try:
        for i, (video, origin, audio) in enumerate(zip(video_frames, original_frames, audio_frames)):
            padded_video[i, :video.shape[0]] = video
            padded_origin[i, :origin.shape[0]] = origin
            padded_audio[i, :audio.shape[0]] = audio
    except Exception as e:
        print(f"Error padding sequences: {e}")
        print(f"Video shape: {video_frames[0].shape}, Origin shape: {original_frames[0].shape}, Audio shape: {audio_frames[0].shape}")
        raise e

    return (padded_origin, padded_video, padded_audio), torch.tensor(labels)