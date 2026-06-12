import streamlit as st
import joblib
import sys

sys.path.append("src")

from predict import predict_winner

encoders = joblib.load("models/encoders.pkl")

st.title("🏏 IPL Match Winner Prediction")

team_list = list(encoders["team1"].classes_)
venue_list = list(encoders["venue"].classes_)

team1 = st.selectbox("Team 1", team_list)

team2 = st.selectbox("Team 2", team_list)

toss_winner = st.selectbox(
    "Toss Winner",
    team_list
)

toss_decision = st.selectbox(
    "Toss Decision",
    ["bat", "field"]
)

venue = st.selectbox(
    "Venue",
    venue_list
)

if st.button("Predict Winner"):

    winner = predict_winner(
        team1,
        team2,
        toss_winner,
        toss_decision,
        venue
    )

    st.success(
        f"Predicted Winner: {winner}"
    )