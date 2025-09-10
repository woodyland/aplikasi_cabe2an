// src/App.js
import React, { useState, useEffect } from "react";
import Camera from "./components/Camera";
import "./App.css";

function App() {
  const [backendStatus, setBackendStatus] = useState(null);
  const [isConnected, setIsConnected] = useState(false);

  // Check backend health on startup
  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const response = await fetch("http://localhost:8000/health");
      const data = await response.json();
      setBackendStatus(data);
      setIsConnected(true);
    } catch (error) {
      console.error("Backend connection failed:", error);
      setIsConnected(false);
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🌶️ Pepper Classification System</h1>
        <p>Real-time pepper genus identification using deep learning</p>

        <div className="status-indicator">
          <div
            className={`status-dot ${
              isConnected ? "connected" : "disconnected"
            }`}
          ></div>
          <span className="status-text">
            {isConnected ? "Backend Connected" : "Backend Disconnected"}
          </span>

          {backendStatus && (
            <div className="backend-info">
              <span
                className={`model-status ${
                  backendStatus.model_loaded ? "loaded" : "not-loaded"
                }`}
              >
                Model:{" "}
                {backendStatus.model_loaded ? "Loaded ✅" : "Not Loaded ❌"}
              </span>
              <span className="classes-count">
                Classes: {backendStatus.classes}
              </span>
            </div>
          )}
        </div>
      </header>

      <main className="app-main">
        {isConnected && backendStatus?.model_loaded ? (
          <Camera />
        ) : (
          <div className="connection-error">
            <h2>Connection Issue</h2>
            {!isConnected ? (
              <div>
                <p>Cannot connect to backend server.</p>
                <p>
                  Please ensure the backend is running on http://localhost:8000
                </p>
              </div>
            ) : !backendStatus?.model_loaded ? (
              <div>
                <p>Backend connected but model is not loaded.</p>
                <p>Please check the server logs for model loading issues.</p>
              </div>
            ) : null}

            <button onClick={checkBackendHealth} className="retry-button">
              🔄 Retry Connection
            </button>
          </div>
        )}
      </main>

      <footer className="app-footer">
        <div className="tech-info">
          <div className="tech-item">
            <strong>Frontend:</strong> React + WebRTC
          </div>
          <div className="tech-item">
            <strong>Backend:</strong> FastAPI + PyTorch
          </div>
          <div className="tech-item">
            <strong>Model:</strong> MobileNet V3 Large
          </div>
          <div className="tech-item">
            <strong>Accuracy:</strong> 95.78% ± 1.02%
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
