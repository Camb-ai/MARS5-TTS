from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3

app = Flask(__name__)
CORS(app)

def init_db():
    conn = sqlite3.connect('ratings.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS ratings
                 (id INTEGER PRIMARY KEY, rating TEXT, feedback TEXT)''')
    conn.commit()
    conn.close()

@app.route('/api/get-audio-samples', methods=['GET'])
def get_audio_samples():
    samples = [
        {'url': 'sample1.mp3'},
        {'url': 'sample2.mp3'}
    ]
    return jsonify(samples)

@app.route('/api/submit-rating', methods=['POST'])
def submit_rating():
    data = request.json
    rating = data.get('rating')
    feedback = data.get('feedback')
    conn = sqlite3.connect('ratings.db')
    c = conn.cursor()
    c.execute("INSERT INTO ratings (rating, feedback) VALUES (?, ?)", (rating, feedback))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Rating submitted successfully'})

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
