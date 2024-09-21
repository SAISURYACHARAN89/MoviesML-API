from flask import Flask, request, jsonify
import joblib
import logging

app = Flask(__name__)

model = joblib.load('Model/model.pkl')
scaler = joblib.load('Model/scaler.pkl')
nameencoder = joblib.load('Model/nameencoder.pkl')
actor1encoder = joblib.load('Model/actor1encoder.pkl')
movieencoder = joblib.load('Model/movieencoder.pkl')
labelencoder_y = joblib.load('Model/labelencoder_y.pkl')

logging.basicConfig(level=logging.DEBUG)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    app.logger.debug(f"Received data: {data}")

    if data is None:
        return jsonify({'error': 'No data received'}), 400

    required_fields = ["movie_title", "director_name", "actor_1_name"]
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400

    # Add your prediction logic here
    movie_title = data['movie_title']
    director_name = data['director_name']
    actor_1_name = data['actor_1_name']

    try:
        # Encode the input data and make predictions
        predict = [
            movieencoder.transform([movie_title])[0],
            nameencoder.transform([director_name])[0],
            actor1encoder.transform([actor_1_name])[0],
        ]

        # Scale the input
        predict = scaler.transform([predict])
        prediction = model.predict(predict)

        result = 'HIT' if prediction[0] == 1 else 'FLOP'
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
