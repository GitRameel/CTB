import React, { useState, useEffect } from "react";
import "./App.css";
import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [symbol, setSymbol] = useState("JUP/USDT");
  const [exchange, setExchange] = useState("mexc");
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const [pineScript, setPineScript] = useState("");
  const [analyzedSymbols, setAnalyzedSymbols] = useState([]);
  const [error, setError] = useState("");
  const [generateLoading, setGenerateLoading] = useState(false);

  useEffect(() => {
    fetchAnalyzedSymbols();
  }, []);

  const fetchAnalyzedSymbols = async () => {
    try {
      const response = await axios.get(`${API}/symbols`);
      setAnalyzedSymbols(response.data.symbols);
    } catch (e) {
      console.error("Error fetching symbols:", e);
    }
  };

  const analyzeSymbol = async () => {
    if (!symbol.trim()) {
      setError("Please enter a trading symbol");
      return;
    }

    setLoading(true);
    setError("");
    setAnalysis(null);

    try {
      const response = await axios.post(`${API}/analyze/${encodeURIComponent(symbol)}?exchange=${exchange}`);
      setAnalysis(response.data);
      await fetchAnalyzedSymbols();
    } catch (e) {
      setError(e.response?.data?.detail || "Failed to analyze symbol");
      console.error("Analysis error:", e);
    } finally {
      setLoading(false);
    }
  };

  const generatePineScript = async () => {
    if (!symbol.trim()) {
      setError("Please analyze a symbol first");
      return;
    }

    setGenerateLoading(true);
    setError("");
    setPineScript("");

    try {
      const response = await axios.post(`${API}/generate-pinescript`, {
        symbol: symbol,
        exchange: exchange,
        lookback_days: 30,
        min_success_rate: 0.65
      });
      
      setPineScript(response.data.pine_script);
    } catch (e) {
      setError(e.response?.data?.detail || "Failed to generate Pine Script");
      console.error("Pine Script generation error:", e);
    } finally {
      setGenerateLoading(false);
    }
  };

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(pineScript);
      alert("Pine Script copied to clipboard!");
    } catch (e) {
      console.error("Failed to copy:", e);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            🚀 TradingView RSI Volume Indicator Generator
          </h1>
          <p className="text-gray-300 text-lg">
            AI-Powered Pattern Recognition for High Volatility Coins
          </p>
          <div className="flex justify-center items-center mt-4 space-x-4">
            <span className="px-3 py-1 bg-green-600 text-white rounded-full text-sm">5m Timeframe</span>
            <span className="px-3 py-1 bg-blue-600 text-white rounded-full text-sm">MEXC & Binance</span>
            <span className="px-3 py-1 bg-purple-600 text-white rounded-full text-sm">0.5% Target</span>
          </div>
        </div>

        {/* Main Controls */}
        <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-lg p-6 mb-8">
          <div className="grid md:grid-cols-3 gap-4 mb-6">
            <div>
              <label className="block text-white text-sm font-medium mb-2">Trading Symbol</label>
              <input
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                placeholder="JUP/USDT, BTC/USDT..."
                className="w-full px-4 py-2 bg-gray-800 text-white rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-white text-sm font-medium mb-2">Exchange</label>
              <select
                value={exchange}
                onChange={(e) => setExchange(e.target.value)}
                className="w-full px-4 py-2 bg-gray-800 text-white rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
              >
                <option value="mexc">MEXC</option>
                <option value="binance">Binance</option>
              </select>
            </div>
            <div className="flex items-end">
              <button
                onClick={analyzeSymbol}
                disabled={loading}
                className="w-full px-6 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium"
              >
                {loading ? "Analyzing..." : "🔍 Analyze Patterns"}
              </button>
            </div>
          </div>

          {error && (
            <div className="mb-4 p-4 bg-red-600 bg-opacity-20 border border-red-500 rounded-lg">
              <p className="text-red-200">{error}</p>
            </div>
          )}
        </div>

        {/* Analysis Results */}
        {analysis && (
          <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-lg p-6 mb-8">
            <h2 className="text-2xl font-bold text-white mb-4">
              📊 Analysis Results for {analysis.symbol}
            </h2>
            
            <div className="grid md:grid-cols-3 gap-4 mb-6">
              <div className="bg-green-600 bg-opacity-20 p-4 rounded-lg border border-green-500">
                <h3 className="text-green-300 font-medium">Total Signals</h3>
                <p className="text-2xl font-bold text-white">{analysis.total_signals}</p>
              </div>
              <div className="bg-blue-600 bg-opacity-20 p-4 rounded-lg border border-blue-500">
                <h3 className="text-blue-300 font-medium">Successful Signals</h3>
                <p className="text-2xl font-bold text-white">{analysis.successful_signals}</p>
              </div>
              <div className="bg-purple-600 bg-opacity-20 p-4 rounded-lg border border-purple-500">
                <h3 className="text-purple-300 font-medium">Success Rate</h3>
                <p className="text-2xl font-bold text-white">{(analysis.success_rate * 100).toFixed(1)}%</p>
              </div>
            </div>

            {analysis.signals && analysis.signals.length > 0 && (
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-white mb-3">Recent Successful Patterns</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-gray-600">
                        <th className="text-left text-gray-300 p-2">Pattern Type</th>
                        <th className="text-left text-gray-300 p-2">RSI</th>
                        <th className="text-left text-gray-300 p-2">Volume Ratio</th>
                        <th className="text-left text-gray-300 p-2">Entry Price</th>
                        <th className="text-left text-gray-300 p-2">Gain %</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analysis.signals.map((signal, index) => (
                        <tr key={index} className="border-b border-gray-700">
                          <td className="text-white p-2">{signal.pattern_type}</td>
                          <td className="text-white p-2">{signal.rsi.toFixed(1)}</td>
                          <td className="text-white p-2">{signal.volume_ratio.toFixed(2)}x</td>
                          <td className="text-white p-2">${signal.entry_price.toFixed(6)}</td>
                          <td className="text-green-400 p-2">+{signal.actual_gain.toFixed(2)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            <button
              onClick={generatePineScript}
              disabled={generateLoading}
              className="w-full px-6 py-3 bg-gradient-to-r from-green-600 to-blue-600 text-white rounded-lg hover:from-green-700 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed font-medium text-lg"
            >
              {generateLoading ? "Generating..." : "🌲 Generate Pine Script for TradingView"}
            </button>
          </div>
        )}

        {/* Pine Script Output */}
        {pineScript && (
          <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-lg p-6 mb-8">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-bold text-white">🌲 Generated Pine Script</h2>
              <button
                onClick={copyToClipboard}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 font-medium"
              >
                📋 Copy to Clipboard
              </button>
            </div>
            
            <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
              <pre className="text-green-400 text-sm whitespace-pre-wrap font-mono">
                {pineScript}
              </pre>
            </div>
            
            <div className="mt-4 p-4 bg-blue-600 bg-opacity-20 border border-blue-500 rounded-lg">
              <h3 className="text-blue-300 font-medium mb-2">📖 How to Use:</h3>
              <ol className="text-blue-200 text-sm space-y-1">
                <li>1. Copy the Pine Script code above</li>
                <li>2. Open TradingView and go to Pine Editor</li>
                <li>3. Paste the code and click "Add to Chart"</li>
                <li>4. The indicator will show RSI + Volume signals optimized for {symbol}</li>
                <li>5. Look for green triangles indicating potential 0.5%+ moves</li>
              </ol>
            </div>
          </div>
        )}

        {/* Previously Analyzed Symbols */}
        {analyzedSymbols.length > 0 && (
          <div className="bg-white bg-opacity-10 backdrop-blur-lg rounded-lg p-6">
            <h2 className="text-2xl font-bold text-white mb-4">📈 Previously Analyzed Symbols</h2>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {analyzedSymbols.map((sym, index) => (
                <div key={index} className="bg-gray-800 bg-opacity-50 p-4 rounded-lg border border-gray-600">
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="text-white font-medium">{sym.symbol}</h3>
                    <span className="text-xs text-gray-400 uppercase">{sym.exchange}</span>
                  </div>
                  <div className="text-sm text-gray-300">
                    <p>Signals: <span className="text-white">{sym.signal_count}</span></p>
                    <p>Success Rate: <span className="text-green-400">{(sym.success_rate * 100).toFixed(1)}%</span></p>
                  </div>
                  <button
                    onClick={() => {
                      setSymbol(sym.symbol);
                      setExchange(sym.exchange);
                    }}
                    className="mt-2 text-xs text-blue-400 hover:text-blue-300"
                  >
                    Use This Symbol →
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="text-center mt-12 text-gray-400">
          <p className="mb-2">🔥 Built for High Volatility Trading • Optimized for 5-minute Timeframes</p>
          <p className="text-sm">⚠️ Trading involves risk. This tool is for educational purposes only.</p>
        </div>
      </div>
    </div>
  );
}

export default App;
