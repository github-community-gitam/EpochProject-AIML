from flask import Flask , request , render_template , jsonify
import pickle
import json
import pandas as pd

app = Flask(__name__)

with open('instance\column.json' , 'r') as file:
    columns = json.load(file)['columns']

with open('model.pkl' , 'rb') as file:
    model = pickle.load(file)

with open('pipeline.pkl' , 'rb') as file:
    pipe = pickle.load(file)

@app.route('/' , methods=['GET'])
def home():
    return render_template('home.html' , predicted_fare=None)

@app.route('/predict' , methods=['POST'])
def predict():
    data = request.form.to_dict().values()
    data = list(map(lambda x: int(x) if x.isdecimal() else x , data))
    df = pd.DataFrame([data] , columns=columns)
    processed_data = pipe.transform(df)
    prediction = model.predict(processed_data)
    return render_template('home.html' , predicted_fare=prediction)

if __name__ == '__main__':
    app.run(debug=True)