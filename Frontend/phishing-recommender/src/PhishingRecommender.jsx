import { useState } from "react";
import axios from "axios";

export default function PhishingRecommender() {
  const [inputDomain, setInputDomain] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [detectionResult, setDetectionResult] = useState(null);

  const handleRecommendation = async () => {
    if (!inputDomain) return;
    setLoading(true);
    setResult(null);
    try {
      const response = await axios.get(
        `http://localhost:8000/recommend?phishing_domain=${inputDomain}`
      );
      setResult(response.data);
    } catch (error) {
      console.error("Error fetching recommendation:", error);
      setResult({ error: "Failed to fetch recommendation. Try again later." });
    }
    setLoading(false);
  };

  const handleDetection = async () => {
    if (!inputDomain) return;
    setLoading(true);
    setDetectionResult(null);
    try {
      const response = await axios.post("http://localhost:8000/predict", {
        url: inputDomain, // Send the input domain for prediction
      });
      setDetectionResult(response.data);
    } catch (error) {
      console.error("Error detecting phishing website:", error);
      setDetectionResult({ error: "Failed to detect. Try again later." });
    }
    setLoading(false);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-6">
      <div className="bg-white shadow-lg rounded-2xl p-6 w-full max-w-md">
        <h1 className="text-2xl font-semibold text-gray-700 text-center mb-4">
          Phishing Website Tool
        </h1>
        <input
          type="text"
          placeholder="Enter website URL..."
          className="w-full p-3 border rounded-lg mb-4"
          value={inputDomain}
          onChange={(e) => setInputDomain(e.target.value)}
        />

        {/* Buttons for recommendation and detection */}
        <div className="flex gap-2">
          <button
            className="w-1/2 bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700"
            onClick={handleRecommendation}
            disabled={loading}
          >
            {loading ? "Searching..." : "Find Legitimate Website"}
          </button>
          <button
            className="w-1/2 bg-red-600 text-white py-2 rounded-lg hover:bg-red-700"
            onClick={handleDetection}
            disabled={loading}
          >
            {loading ? "Checking..." : "Check if Phishing"}
          </button>
        </div>

        {/* Display Legitimate Website Recommendation */}
        {result?.recommended_legit && (
          <div className="mt-4 p-4 bg-green-100 text-green-700 rounded-lg">
            <p>
              <strong>Are you Looking for </strong>
              <a
                href={`https://${result.recommended_legit}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 underline"
              >
                {result.recommended_legit}
              </a>
            </p>
          </div>
        )}

        {/* Display Phishing Detection Result */}
        {detectionResult?.prediction && (
          <div
            className={`mt-4 p-4 rounded-lg ${
              detectionResult.prediction === "Phishing"
                ? "bg-red-100 text-red-700"
                : "bg-green-100 text-green-700"
            }`}
          >
            <p>
              <strong>Detection Result: </strong>
              {detectionResult.prediction}
            </p>
          </div>
        )}

        {/* Display Errors */}
        {result?.error && (
          <div className="mt-4 p-4 bg-red-100 text-red-700 rounded-lg">
            <p>{result.error}</p>
          </div>
        )}
        {detectionResult?.error && (
          <div className="mt-4 p-4 bg-red-100 text-red-700 rounded-lg">
            <p>{detectionResult.error}</p>
          </div>
        )}
      </div>
    </div>
  );
}
