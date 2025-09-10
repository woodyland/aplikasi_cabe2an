// src/components/Camera.js
import React, { useRef, useEffect, useState } from "react";
import axios from "axios";

const Camera = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Start camera stream
  const startCamera = async () => {
    try {
      // Try back camera first
      const constraints = {
        video: {
          width: 640,
          height: 480,
          facingMode: { exact: "environment" },
        },
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.style.transform = "none"; // No mirror untuk back camera
        setIsStreaming(true);
        setError(null);
      }
    } catch (err) {
      // Fallback ke front camera dengan mirror fix
      try {
        const fallbackStream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" },
        });

        if (videoRef.current) {
          videoRef.current.srcObject = fallbackStream;
          videoRef.current.style.transform = "scaleX(-1)"; // Mirror fix
          setIsStreaming(true);
          setError(null);
        }
      } catch (fallbackErr) {
        setError("Camera access denied.");
      }
    }
  };

  // Stop camera stream
  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject;
      const tracks = stream.getTracks();
      tracks.forEach((track) => track.stop());
      videoRef.current.srcObject = null;
      setIsStreaming(false);
    }
  };

  // Capture image and predict
  const captureAndPredict = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    setIsLoading(true);

    try {
      const canvas = canvasRef.current;
      const video = videoRef.current;

      // Set canvas size to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Draw current video frame to canvas
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0);

      // Convert canvas to blob
      canvas.toBlob(
        async (blob) => {
          const formData = new FormData();
          formData.append("file", blob, "capture.jpg");

          try {
            const response = await axios.post(
              "http://localhost:8000/predict",
              formData,
              {
                headers: {
                  "Content-Type": "multipart/form-data",
                },
              }
            );

            if (response.data.error) {
              setError(response.data.error);
            } else {
              setPrediction(response.data.prediction);
              setConfidence(response.data.confidence);
              setError(null);
            }
          } catch (err) {
            setError("Prediction failed. Please try again.");
            console.error("Prediction error:", err);
          }

          setIsLoading(false);
        },
        "image/jpeg",
        0.8
      );
    } catch (err) {
      setError("Capture failed. Please try again.");
      setIsLoading(false);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <div className="camera-container">
      <div className="camera-section">
        <h2>Pepper Classification Camera</h2>

        {error && <div className="error-message">⚠️ {error}</div>}

        <div className="video-container">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="video-stream"
          />
          <canvas ref={canvasRef} style={{ display: "none" }} />
        </div>

        <div className="camera-controls">
          {!isStreaming ? (
            <button onClick={startCamera} className="btn btn-primary">
              📷 Start Camera
            </button>
          ) : (
            <div className="streaming-controls">
              <button
                onClick={captureAndPredict}
                disabled={isLoading}
                className="btn btn-success"
              >
                {isLoading ? "🔄 Processing..." : "📸 Capture & Classify"}
              </button>
              <button onClick={stopCamera} className="btn btn-secondary">
                🛑 Stop Camera
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="results-section">
        <h3>Classification Results</h3>

        {prediction ? (
          <div className="prediction-result">
            <div className="prediction-class">
              <strong>Predicted Class:</strong>
              <span className="class-name">{prediction}</span>
            </div>

            <div className="confidence-score">
              <strong>Confidence:</strong>
              <div className="confidence-bar">
                <div
                  className="confidence-fill"
                  style={{ width: `${confidence}%` }}
                ></div>
                <span className="confidence-text">
                  {confidence.toFixed(1)}%
                </span>
              </div>
            </div>

            <div className="confidence-level">
              {confidence >= 80
                ? "✅ High Confidence"
                : confidence >= 60
                ? "⚠️ Medium Confidence"
                : "❌ Low Confidence"}
            </div>
          </div>
        ) : (
          <div className="no-prediction">
            📷 Capture an image to see classification results
          </div>
        )}
      </div>
    </div>
  );
};

export default Camera;
