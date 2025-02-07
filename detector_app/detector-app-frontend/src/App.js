import logo from './logo.svg';
import './App.css';
import React, { useState } from 'react';

function App() {
  const [headline, setHeadline] = useState('');
  const [result, setResult] = useState('');
  
  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ headline })
      });
      
      const data = await response.json();
      setResult(data.prediction);
    } catch (error) {
      console.error(error);
      setResult('Error: could not get prediction');
    }
  };

  return (
    <div style={{ margin: '2rem'}}>
      <h1>Fake News Detector</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Headline:
          <input
            type='text'
            value={headline}
            onChange={(e) => setHeadline(e.target.value)}
          />
        </label>
        <button type='submit'>Predict</button>
      </form>
      {result && <p>Prediction: {result}</p>}
    </div>
  );
}

export default App;
