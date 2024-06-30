import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [audioSamples, setAudioSamples] = useState([]);
  const [rating, setRating] = useState(null);
  const [feedback, setFeedback] = useState('');

  useEffect(() => {
    axios.get('/api/get-audio-samples')
      .then(response => setAudioSamples(response.data))
      .catch(error => console.error(error));
  }, []);

  const submitRating = () => {
    axios.post('/api/submit-rating', { rating, feedback })
      .then(response => console.log(response.data))
      .catch(error => console.error(error));
  };

  return (
    <div>
      <h1>MARS5 Human Preference Ratings</h1>
      {audioSamples.map((sample, index) => (
        <audio key={index} controls>
          <source src={sample.url} type="audio/mpeg" />
        </audio>
      ))}
      <div>
        <button onClick={() => setRating('model')}>Model</button>
        <button onClick={() => setRating('ground_truth')}>Ground Truth</button>
      </div>
      <div>
        <textarea
          placeholder="Optional feedback"
          value={feedback}
          onChange={(e) => setFeedback(e.target.value)}
        />
      </div>
      <button onClick={submitRating}>Submit Rating</button>
    </div>
  );
}

export default App;
