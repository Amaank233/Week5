from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn import datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


iris = datasets.load_iris()
data = iris.data
target = iris.target


df = pd.DataFrame(data, columns=iris.feature_names)
df['target'] = target


scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.drop('target', axis=1))


lr = LogisticRegression(max_iter=1000)
lr.fit(scaled_data, df['target'])

joblib.dump(lr, 'model.pkl')


model = joblib.load('model.pkl')


app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return df.to_html()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict(np.array([data['features']]))
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(port=5000, debug=True)

