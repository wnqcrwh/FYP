import os
import pandas as pd
from moviepy.editor import VideoFileClip

csv_path = "MELD/test.csv"
video_root = "MELD/test"
clean_csv_path = "MELD/test_cleaned.csv"

df = pd.read_csv(csv_path)
valid_rows = []

for i, row in df.iterrows():
    filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
    video_path = os.path.join(video_root, filename)
    
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        continue
    
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        clip.reader.close()
        clip.audio.reader.close_proc()
        if duration <= 0:
            print(f"Invalid duration: {video_path}")
            continue
    except Exception as e:
        print(f"Cannot load: {video_path} ({e})")
        continue
    
    valid_rows.append(row)


pd.DataFrame(valid_rows).to_csv(clean_csv_path, index=False)
print(f"Saved cleaned CSV to {clean_csv_path}")