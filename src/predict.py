import joblib

model = joblib.load("models/model.pkl")
encoders = joblib.load("models/encoders.pkl")


def predict_winner(
    team1,
    team2,
    toss_winner,
    toss_decision,
    venue
):

    data = [[
        encoders["team1"].transform([team1])[0],
        encoders["team2"].transform([team2])[0],
        encoders["toss_winner"].transform([toss_winner])[0],
        encoders["toss_decision"].transform([toss_decision])[0],
        encoders["venue"].transform([venue])[0]
    ]]

    prediction = model.predict(data)

    winner = encoders["winner"].inverse_transform(prediction)

    return winner[0]