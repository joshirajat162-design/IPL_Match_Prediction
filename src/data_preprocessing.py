import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data():

    matches = pd.read_csv("data/matches.csv")

    matches = matches[
        [
            "team1",
            "team2",
            "toss_winner",
            "toss_decision",
            "venue",
            "winner"
        ]
    ]

    matches.dropna(inplace=True)

    encoders = {}

    columns = [
        "team1",
        "team2",
        "toss_winner",
        "toss_decision",
        "venue",
        "winner"
    ]

    for col in columns:
        le = LabelEncoder()
        matches[col] = le.fit_transform(matches[col])
        encoders[col] = le

    return matches, encoders