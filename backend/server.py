from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
import aiohttp
import asyncio
import pandas as pd
import numpy as np
from statistics import mean
import json

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="TradingView Indicator Generator", description="RSI Volume Price Action Pattern Analysis")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define Models
class TradingSymbol(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    exchange: str
    base_asset: str
    quote_asset: str
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

class KlineData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    exchange: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    quote_volume: float
    interval: str = "5m"  # Default to 5 minute

class TechnicalIndicators(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    timestamp: datetime
    rsi: float
    volume_sma: float
    volume_ratio: float
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    price_change_5m: Optional[float] = None
    price_change_15m: Optional[float] = None

class PatternSignal(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    timestamp: datetime
    pattern_type: str
    signal_strength: float
    entry_price: float
    target_price: float
    expected_return: float
    rsi_value: float
    volume_ratio: float
    success_probability: float

class PineScriptRequest(BaseModel):
    symbol: str
    exchange: str = "mexc"
    lookback_days: int = 30
    min_success_rate: float = 0.65

# Exchange API Classes
class MEXCConnector:
    BASE_URL = "https://api.mexc.com"
    
    @staticmethod
    async def get_kline_data(symbol: str, interval: str = "5m", limit: int = 1000):
        """Fetch kline data from MEXC"""
        try:
            url = f"{MEXCConnector.BASE_URL}/api/v3/klines"
            params = {
                "symbol": symbol.replace("/", ""),
                "interval": interval,
                "limit": limit
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        klines = []
                        for k in data:
                            klines.append({
                                "timestamp": datetime.fromtimestamp(int(k[0]) / 1000),
                                "open": float(k[1]),
                                "high": float(k[2]),
                                "low": float(k[3]),
                                "close": float(k[4]),
                                "volume": float(k[5]),
                                "quote_volume": float(k[7])
                            })
                        return klines
                    else:
                        logger.error(f"MEXC API error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching MEXC data: {str(e)}")
            return []

class BinanceConnector:
    BASE_URL = "https://api.binance.com"
    
    @staticmethod
    async def get_kline_data(symbol: str, interval: str = "5m", limit: int = 1000):
        """Fetch kline data from Binance"""
        try:
            url = f"{BinanceConnector.BASE_URL}/api/v3/klines"
            params = {
                "symbol": symbol.replace("/", ""),
                "interval": interval,
                "limit": limit
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        klines = []
                        for k in data:
                            klines.append({
                                "timestamp": datetime.fromtimestamp(int(k[0]) / 1000),
                                "open": float(k[1]),
                                "high": float(k[2]),
                                "low": float(k[3]),
                                "close": float(k[4]),
                                "volume": float(k[5]),
                                "quote_volume": float(k[7])
                            })
                        return klines
                    else:
                        logger.error(f"Binance API error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Error fetching Binance data: {str(e)}")
            return []

# Technical Analysis Functions
def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """Calculate RSI values"""
    if len(prices) < period + 1:
        return [50.0] * len(prices)
    
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    
    avg_gain = mean(gains[:period])
    avg_loss = mean(losses[:period])
    
    rsi_values = []
    
    for i in range(period, len(deltas)):
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        rsi_values.append(rsi)
        
        # Update averages
        if i < len(deltas) - 1:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    # Pad with initial values
    return [50.0] * (period + 1) + rsi_values

def find_support_resistance(highs: List[float], lows: List[float], lookback: int = 20) -> tuple:
    """Find support and resistance levels"""
    if len(highs) < lookback or len(lows) < lookback:
        return None, None
    
    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    
    # Simple approach: use recent high/low levels that were tested multiple times
    resistance = max(recent_highs)
    support = min(recent_lows)
    
    return support, resistance

def analyze_patterns(klines: List[dict]) -> List[dict]:
    """Analyze price patterns for signals"""
    if len(klines) < 50:
        return []
    
    closes = [k["close"] for k in klines]
    volumes = [k["volume"] for k in klines]
    highs = [k["high"] for k in klines]
    lows = [k["low"] for k in klines]
    
    rsi_values = calculate_rsi(closes)
    volume_sma = pd.Series(volumes).rolling(window=20).mean().tolist()
    
    signals = []
    
    for i in range(30, len(klines)):
        current_rsi = rsi_values[i] if i < len(rsi_values) else 50
        current_volume = volumes[i]
        avg_volume = volume_sma[i] if volume_sma[i] else current_volume
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        support, resistance = find_support_resistance(highs[:i+1], lows[:i+1])
        
        # Pattern 1: RSI Oversold + Volume Spike + Near Support
        if (current_rsi < 30 and volume_ratio > 1.5 and 
            support and closes[i] <= support * 1.02):
            
            # Check if this led to 0.5% gain in next 5-15 periods
            future_gain = 0
            if i + 15 < len(closes):
                max_future_price = max(closes[i+1:i+16])
                future_gain = (max_future_price - closes[i]) / closes[i] * 100
            
            if future_gain >= 0.5:
                signals.append({
                    "timestamp": klines[i]["timestamp"],
                    "pattern_type": "RSI_OVERSOLD_VOLUME_SPIKE",
                    "entry_price": closes[i],
                    "rsi": current_rsi,
                    "volume_ratio": volume_ratio,
                    "support_level": support,
                    "actual_gain": future_gain,
                    "signal_strength": min(100, (volume_ratio * 20) + (40 - current_rsi))
                })
        
        # Pattern 2: RSI Divergence + Volume Confirmation
        if i >= 10:
            price_trend = closes[i] - closes[i-10]
            rsi_trend = rsi_values[i] - rsi_values[i-10] if i < len(rsi_values) and i-10 < len(rsi_values) else 0
            
            # Bullish divergence: price down, RSI up, volume spike
            if price_trend < 0 and rsi_trend > 0 and volume_ratio > 1.3:
                future_gain = 0
                if i + 15 < len(closes):
                    max_future_price = max(closes[i+1:i+16])
                    future_gain = (max_future_price - closes[i]) / closes[i] * 100
                
                if future_gain >= 0.5:
                    signals.append({
                        "timestamp": klines[i]["timestamp"],
                        "pattern_type": "BULLISH_DIVERGENCE",
                        "entry_price": closes[i],
                        "rsi": current_rsi,
                        "volume_ratio": volume_ratio,
                        "actual_gain": future_gain,
                        "signal_strength": min(100, abs(rsi_trend) * 10 + volume_ratio * 15)
                    })
    
    return signals

# API Routes
@api_router.get("/")
async def root():
    return {"message": "TradingView Indicator Generator API"}

@api_router.post("/analyze/{symbol}")
async def analyze_symbol(symbol: str, exchange: str = "mexc"):
    """Analyze a trading symbol for patterns"""
    try:
        # Fetch data from exchange
        if exchange.lower() == "mexc":
            klines = await MEXCConnector.get_kline_data(symbol)
        else:
            klines = await BinanceConnector.get_kline_data(symbol)
        
        if not klines:
            raise HTTPException(status_code=404, detail="No data found for symbol")
        
        # Store kline data
        for kline in klines:
            kline_doc = KlineData(
                symbol=symbol,
                exchange=exchange,
                timestamp=kline["timestamp"],
                open_price=kline["open"],
                high_price=kline["high"],
                low_price=kline["low"],
                close_price=kline["close"],
                volume=kline["volume"],
                quote_volume=kline["quote_volume"]
            )
            await db.kline_data.update_one(
                {"symbol": symbol, "timestamp": kline["timestamp"], "exchange": exchange},
                {"$set": kline_doc.dict()},
                upsert=True
            )
        
        # Analyze patterns
        signals = analyze_patterns(klines)
        
        # Store signals
        for signal in signals:
            signal_doc = PatternSignal(
                symbol=symbol,
                timestamp=signal["timestamp"],
                pattern_type=signal["pattern_type"],
                signal_strength=signal["signal_strength"],
                entry_price=signal["entry_price"],
                target_price=signal["entry_price"] * 1.005,  # 0.5% target
                expected_return=0.5,
                rsi_value=signal["rsi"],
                volume_ratio=signal["volume_ratio"],
                success_probability=signal["actual_gain"] / 0.5 if signal["actual_gain"] >= 0.5 else 0
            )
            await db.pattern_signals.update_one(
                {"symbol": symbol, "timestamp": signal["timestamp"], "pattern_type": signal["pattern_type"]},
                {"$set": signal_doc.dict()},
                upsert=True
            )
        
        return {
            "symbol": symbol,
            "exchange": exchange,
            "total_signals": len(signals),
            "successful_signals": len([s for s in signals if s["actual_gain"] >= 0.5]),
            "success_rate": len([s for s in signals if s["actual_gain"] >= 0.5]) / len(signals) if signals else 0,
            "signals": signals[:10]  # Return latest 10 signals
        }
        
    except Exception as e:
        logger.error(f"Error analyzing symbol {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/generate-pinescript")
async def generate_pine_script(request: PineScriptRequest):
    """Generate Pine Script code based on analyzed patterns"""
    try:
        # Get successful patterns for the symbol
        signals = await db.pattern_signals.find({
            "symbol": request.symbol,
            "success_probability": {"$gte": request.min_success_rate}
        }).to_list(1000)
        
        if not signals:
            raise HTTPException(status_code=404, detail="No successful patterns found for symbol")
        
        # Analyze pattern characteristics
        rsi_oversold_signals = [s for s in signals if s["pattern_type"] == "RSI_OVERSOLD_VOLUME_SPIKE"]
        divergence_signals = [s for s in signals if s["pattern_type"] == "BULLISH_DIVERGENCE"]
        
        avg_rsi_oversold = mean([s["rsi_value"] for s in rsi_oversold_signals]) if rsi_oversold_signals else 30
        avg_volume_ratio = mean([s["volume_ratio"] for s in signals])
        
        # Generate Pine Script
        pine_script = f'''// TradingView Pine Script - RSI Volume Price Action Indicator
// Generated for {request.symbol} based on {len(signals)} successful patterns
// Min Success Rate: {request.min_success_rate*100}%

//@version=5
indicator("RSI Volume Price Action - {request.symbol}", shorttitle="RVPA-{request.symbol.split('/')[0]}", overlay=false)

// Input Parameters
rsi_length = input.int(14, title="RSI Length", minval=1)
rsi_oversold = input.float({avg_rsi_oversold:.1f}, title="RSI Oversold Level", minval=1, maxval=50)
volume_multiplier = input.float({avg_volume_ratio:.1f}, title="Volume Spike Multiplier", minval=1.0, step=0.1)
lookback_periods = input.int(20, title="Support/Resistance Lookback", minval=5)

// Calculate RSI
rsi = ta.rsi(close, rsi_length)

// Calculate Volume Ratio
volume_sma = ta.sma(volume, 20)
volume_ratio = volume / volume_sma

// Support and Resistance
support = ta.lowest(low, lookback_periods)
resistance = ta.highest(high, lookback_periods)

// Pattern Conditions
oversold_condition = rsi < rsi_oversold and volume_ratio > volume_multiplier and close <= support * 1.02
divergence_condition = rsi > rsi[10] and close < close[10] and volume_ratio > (volume_multiplier * 0.8)

// Signals
buy_signal = oversold_condition or divergence_condition

// Plotting
hline(70, "Overbought", color=color.red, linestyle=hline.style_dashed)
hline(30, "Oversold", color=color.green, linestyle=hline.style_dashed)
hline(50, "Midline", color=color.gray, linestyle=hline.style_dotted)

plot(rsi, title="RSI", color=color.blue, linewidth=2)
plot(volume_ratio * 10, title="Volume Ratio (x10)", color=color.orange, linewidth=1)

// Signal Visualization
plotshape(buy_signal, title="Buy Signal", location=location.bottom, color=color.lime, style=shape.triangleup, size=size.small)

// Background coloring for signal strength
bgcolor(buy_signal ? color.new(color.green, 90) : na, title="Signal Background")

// Alerts
alertcondition(buy_signal, title="RSI Volume Price Action Signal", message="Potential 0.5% move detected for {request.symbol}")

// Statistics Panel
var table stats_table = table.new(position.top_right, 2, 4, bgcolor=color.white, border_width=1)
if barstate.islast
    table.cell(stats_table, 0, 0, "Success Rate", text_color=color.black, text_size=size.small)
    table.cell(stats_table, 1, 0, "{(len([s for s in signals if s['success_probability'] >= request.min_success_rate]) / len(signals) * 100):.1f}%", text_color=color.green, text_size=size.small)
    table.cell(stats_table, 0, 1, "Total Signals", text_color=color.black, text_size=size.small)
    table.cell(stats_table, 1, 1, "{len(signals)}", text_color=color.blue, text_size=size.small)
    table.cell(stats_table, 0, 2, "RSI Threshold", text_color=color.black, text_size=size.small)
    table.cell(stats_table, 1, 2, str.tostring(rsi_oversold, "#.#"), text_color=color.purple, text_size=size.small)
    table.cell(stats_table, 0, 3, "Volume Multi", text_color=color.black, text_size=size.small)
    table.cell(stats_table, 1, 3, str.tostring(volume_multiplier, "#.#"), text_color=color.orange, text_size=size.small)
'''
        
        return {
            "symbol": request.symbol,
            "pine_script": pine_script,
            "total_patterns": len(signals),
            "success_rate": len([s for s in signals if s["success_probability"] >= request.min_success_rate]) / len(signals) if signals else 0,
            "avg_rsi_level": avg_rsi_oversold,
            "avg_volume_ratio": avg_volume_ratio,
            "pattern_types": list(set([s["pattern_type"] for s in signals]))
        }
        
    except Exception as e:
        logger.error(f"Error generating Pine Script: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/symbols")
async def get_analyzed_symbols():
    """Get list of analyzed symbols"""
    try:
        pipeline = [
            {"$group": {
                "_id": {
                    "symbol": "$symbol",
                    "exchange": "$exchange"
                },
                "signal_count": {"$sum": 1},
                "success_rate": {"$avg": "$success_probability"},
                "last_analysis": {"$max": "$timestamp"}
            }},
            {"$sort": {"signal_count": -1}}
        ]
        
        results = await db.pattern_signals.aggregate(pipeline).to_list(100)
        
        symbols = []
        for result in results:
            symbols.append({
                "symbol": result["_id"]["symbol"],
                "exchange": result["_id"]["exchange"],
                "signal_count": result["signal_count"],
                "success_rate": result["success_rate"],
                "last_analysis": result["last_analysis"]
            })
        
        return {"symbols": symbols}
        
    except Exception as e:
        logger.error(f"Error getting symbols: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
