import pandas as pd
import os

input_csv="MELD/dev_sent_emo.csv"
output_csv="MELD/dev.csv"
LABEL_MAP_MELD = {
    "Emotion": {
        "neutral": 0,
        "joy": 1,
        "sadness": 2,
        "anger": 3,
        "surprise": 4,
        "fear": 5,
        "disgust": 6
    },
    "Sentiment": {
        "positive": 0,
        "neutral": 1,
        "negative": 2
    }
}


def convert_labels(emotion: str, sentiment: str):
    """情绪和情感标签转换"""
    emotion_label = LABEL_MAP_MELD["Emotion"].get(emotion, -1)
    sentiment_label = LABEL_MAP_MELD["Sentiment"].get(sentiment, -1)
    return emotion_label, sentiment_label

df=pd.read_csv(input_csv)
#convert emotion and sentiment labels to numerical values
df[["Emotion", "Sentiment"]] = df.apply(
    lambda row: pd.Series(convert_labels(row["Emotion"], row["Sentiment"])),
    axis=1
)
df.to_csv(output_csv, index=False)
print("Labels converted and saved to", output_csv)
# for index,row in df.iterrows():
#     emotion_label, sentiment_label = convert_labels(row["Emotion"], row["Sentiment"])
#     row["Emotion"] = emotion_label
#     row["Sentiment"] = sentiment_label

#     row_df = row.to_frame().T
#     output_file = os.path.join(output_dir, f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.csv")
#     row_df.to_csv(output_file, index=False)

