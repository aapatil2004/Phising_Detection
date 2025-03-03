import { useState } from "react";
import axios from "axios";

export default function PhishingRecommender() {
  const [inputDomain, setInputDomain] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!inputDomain) return;
    setLoading(true);
    try {
      const response = await axios.get(
        `http://localhost:8000/recommend?phishing_domain=${inputDomain}`
      );
      setResult(response.data);
    } catch (error) {
      console.error("Error fetching recommendation:", error);
    }
    setLoading(false);
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-6">
      <div className="bg-white shadow-lg rounded-2xl p-6 w-full max-w-md">
        <h1 className="text-2xl font-semibold text-gray-700 text-center mb-4">
          Phishing Website Recommender
        </h1>
        <input
          type="text"
          placeholder="Enter phishing domain..."
          className="w-full p-3 border rounded-lg mb-4"
          value={inputDomain}
          onChange={(e) => setInputDomain(e.target.value)}
        />
        <button
          className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700"
          onClick={handleSearch}
          disabled={loading}
        >
          {loading ? "Searching..." : "Find Legitimate Website"}
        </button>

        {result && (
          <div className="mt-4 p-4 bg-green-100 text-green-700 rounded-lg">
            <p>
              <strong>Are you Looking for </strong>
              <a
                href={`https://${result.recommended}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 underline"
              >
                {result.recommended}
              </a>
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
