import requests
import unittest
import json
import sys
from datetime import datetime

class TradingViewIndicatorAPITest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TradingViewIndicatorAPITest, self).__init__(*args, **kwargs)
        self.base_url = "https://00e21891-7197-475c-8b32-e705ed474680.preview.emergentagent.com/api"
        self.test_symbol = "JUPUSDT"
        self.test_exchange = "mexc"
        
    def test_01_health_check(self):
        """Test API health check endpoint"""
        print("\n🔍 Testing API health check...")
        response = requests.get(f"{self.base_url}/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        print("✅ API health check passed")
        
    def test_02_analyze_symbol(self):
        """Test symbol analysis endpoint"""
        print("\n🔍 Testing symbol analysis...")
        response = requests.post(f"{self.base_url}/analyze/{self.test_symbol}?exchange={self.test_exchange}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify response structure
        self.assertIn("symbol", data)
        self.assertIn("exchange", data)
        self.assertIn("total_signals", data)
        self.assertIn("successful_signals", data)
        self.assertIn("success_rate", data)
        
        # Store for later tests
        self.analysis_result = data
        
        print(f"✅ Symbol analysis passed - Found {data['total_signals']} signals with {data['successful_signals']} successful")
        print(f"   Success rate: {data['success_rate'] * 100:.1f}%")
        
    def test_03_generate_pinescript(self):
        """Test Pine Script generation endpoint"""
        print("\n🔍 Testing Pine Script generation...")
        payload = {
            "symbol": self.test_symbol,
            "exchange": self.test_exchange,
            "lookback_days": 30,
            "min_success_rate": 0.65
        }
        
        response = requests.post(f"{self.base_url}/generate-pinescript", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify response structure
        self.assertIn("symbol", data)
        self.assertIn("pine_script", data)
        self.assertIn("total_patterns", data)
        self.assertIn("success_rate", data)
        
        # Verify Pine Script content
        pine_script = data["pine_script"]
        self.assertIn("//@version=5", pine_script)
        self.assertIn("indicator(", pine_script)
        self.assertIn("rsi =", pine_script)
        
        print(f"✅ Pine Script generation passed - Generated {len(pine_script.splitlines())} lines of code")
        print(f"   Total patterns: {data['total_patterns']}")
        
    def test_04_get_symbols(self):
        """Test getting list of analyzed symbols"""
        print("\n🔍 Testing symbols list endpoint...")
        response = requests.get(f"{self.base_url}/symbols")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Verify response structure
        self.assertIn("symbols", data)
        symbols = data["symbols"]
        
        # After our analysis, we should have at least one symbol
        self.assertGreaterEqual(len(symbols), 1)
        
        # Verify symbol data structure
        if symbols:
            symbol = symbols[0]
            self.assertIn("symbol", symbol)
            self.assertIn("exchange", symbol)
            self.assertIn("signal_count", symbol)
            self.assertIn("success_rate", symbol)
            
        print(f"✅ Symbols list endpoint passed - Found {len(symbols)} analyzed symbols")

def run_tests():
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add tests in order
    suite.addTest(TradingViewIndicatorAPITest('test_01_health_check'))
    suite.addTest(TradingViewIndicatorAPITest('test_02_analyze_symbol'))
    suite.addTest(TradingViewIndicatorAPITest('test_03_generate_pinescript'))
    suite.addTest(TradingViewIndicatorAPITest('test_04_get_symbols'))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_tests())
