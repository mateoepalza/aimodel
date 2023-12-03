from flask import Flask, request, jsonify
import json
import joblib
import numpy as np
from sklearn import preprocessing

app = Flask(__name__)

def select_winning_team(probability):
    prob_lst=[round(probability[0][i],3) for i in range(2)]
    if (prob_lst[0]>prob_lst[1]):
        out=0
    elif (prob_lst[0]<prob_lst[1]):
        out=1
    elif (prob_lst[0]==prob_lst[1]):
        out=2
    return out,prob_lst

def predict(features):
    loaded_model = joblib.load('model.joblib')
    return loaded_model.predict(features)

def encode_fields(year, team_1, team_2, stadium):
    encoder = preprocessing.LabelEncoder()
    encoder.classes_ = np.load('classes.npy', allow_pickle = True)

    team_1_num=encoder.transform([team_1])[0]
    team_2_num=encoder.transform([team_2])[0]
    stadium_num=encoder.transform([stadium])[0]

    return np.array([[year,stadium_num,team_1_num,team_2_num]])


@app.route('/predict', methods=['POST'])
def predict_match():
    # Get the data
    data = request.json

    # Get the fields
    year = data['year']
    team_1 = data['homeTeam']
    team_2 = data['awayTeam']
    stadium = data['stadium']

    team_lst = [team_1, team_2]

    # Encode the fields
    feature = encode_fields(year, team_1, team_2, stadium)

    # Predict
    prediction = predict(feature)

    # Select the winning team
    win, _ = select_winning_team(prediction)

    try:
        result = f"{team_1} vs {team_2} \n {team_lst[win]} wins 🏆⚽🎯\n"
    except IndexError:
        result = f"{team_1} vs {team_2} \n  Match Draw ⚽⚽⚽\n"

    print(f"This is the result: {result}")

    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)