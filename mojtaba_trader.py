# mojtaba_trader_pump_dump_full.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import json
import warnings
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# ===================== CONFIG ===============================
st.set_page_config(
    page_title="Mojtaba_D Trader Pro ULTIMATE v2.0 - PUMP & DUMP EDITION",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# ===================== لیست ۱۵ ارز پامپ و دامپ ===============================
SYMBOLS = [
    # 🚀 پامپ‌خیزترین ارزها (نوسان ۱۰۰%+ در هفته)
    "PEPEUSDT",     # نوسان روزانه ۵۰-۱۰۰%
    "SHIBUSDT",     # حجم میلیاردی، پامپ‌های سریع
    "FLOKIUSDT",    # نوسان‌گیر قوی مم‌کوین
    "WIFUSDT",      # جدیدترین و پامپ‌خیزترین
    "BONKUSDT",     # نوسان‌گیر سولانا اکوسیستم
    
    # 📈 آلت‌کوین‌های نوسانی بالا
    "SOLUSDT",      # پادشاه نوسان‌گیری
    "DOGEUSDT",     # پامپ‌های لحظه‌ای
    "ADAUSDT",      # نوسان‌گیر قدیمی اما قوی
    "AVAXUSDT",     # نوسان خوب + حجم بالا
    "MATICUSDT",    # پامپ‌های مکرر
    
    # 💥 ارزهای با پتانسیل انفجار
    "NEARUSDT",     # نوسان بالا + اخبار قوی
    "APTUSDT",      # پامپ‌های غافلگیرانه
    "ARBUSDT",      # نوسان‌گیر لایه ۲
    "OPUSDT",       # پامپ‌های سریع
    "RNDRUSDT"      # AI سکه نوسانی
]

# ===================== سیستم ۳ صرافی ===============================
class TripleExchangeConnector:
    def __init__(self):
        self.bybit_prices = {}
        self.binance_prices = {}
        self.okx_prices = {}
        self.best_prices = {}
        self.sources = {}
        self.last_update = {}
        self.volumes = {}
        
    def get_bybit_data(self, symbol):
        """داده‌های کامل از Bybit"""
        try:
            url = "https://api.bybit.com/v5/market/tickers"
            params = {"category": "spot", "symbol": symbol}
            response = requests.get(url, params=params, timeout=3)
            if response.status_code == 200:
                data = response.json()
                if data.get('retCode') == 0:
                    ticker = data['result']['list'][0]
                    return {
                        'price': float(ticker['lastPrice']),
                        'volume': float(ticker['volume24h']),
                        'high': float(ticker['highPrice24h']),
                        'low': float(ticker['lowPrice24h'])
                    }
        except:
            return None
        return None
    
    def get_binance_data(self, symbol):
        """داده‌های کامل از Binance"""
        try:
            price_url = "https://api.binance.com/api/v3/ticker/price"
            price_response = requests.get(price_url, params={"symbol": symbol}, timeout=3)
            
            ticker_url = "https://api.binance.com/api/v3/ticker/24hr"
            ticker_response = requests.get(ticker_url, params={"symbol": symbol}, timeout=3)
            
            if price_response.status_code == 200 and ticker_response.status_code == 200:
                price_data = price_response.json()
                ticker_data = ticker_response.json()
                
                return {
                    'price': float(price_data['price']),
                    'volume': float(ticker_data['volume']),
                    'high': float(ticker_data['highPrice']),
                    'low': float(ticker_data['lowPrice'])
                }
        except:
            return None
        return None
    
    def get_okx_data(self, symbol):
        """داده‌های کامل از OKX"""
        try:
            okx_symbol = symbol.replace('USDT', '-USDT')
            url = "https://www.okx.com/api/v5/market/ticker"
            params = {"instId": okx_symbol}
            response = requests.get(url, params=params, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '0':
                    ticker = data['data'][0]
                    return {
                        'price': float(ticker['last']),
                        'volume': float(ticker['vol24h']),
                        'high': float(ticker['high24h']),
                        'low': float(ticker['low24h'])
                    }
        except:
            return None
        return None
    
    def get_best_price_with_volume(self, symbol):
        """بهترین قیمت با اطلاعات حجم"""
        prices = {}
        volumes = {}
        
        # Bybit
        bybit_data = self.get_bybit_data(symbol)
        if bybit_data:
            prices['Bybit'] = bybit_data['price']
            volumes['Bybit'] = bybit_data['volume']
        
        # Binance
        binance_data = self.get_binance_data(symbol)
        if binance_data:
            prices['Binance'] = binance_data['price']
            volumes['Binance'] = binance_data['volume']
        
        # OKX
        okx_data = self.get_okx_data(symbol)
        if okx_data:
            prices['OKX'] = okx_data['price']
            volumes['OKX'] = okx_data['volume']
        
        if prices:
            best_source = min(prices, key=prices.get)
            best_price = prices[best_source]
            best_volume = volumes.get(best_source, 0)
            
            self.best_prices[symbol] = best_price
            self.sources[symbol] = best_source
            self.volumes[symbol] = best_volume
            self.last_update[symbol] = datetime.now()
            
            return {
                'price': best_price,
                'source': best_source,
                'volume': best_volume,
                'all_prices': prices,
                'all_volumes': volumes,
                'available_exchanges': list(prices.keys()),
                'exchange_count': len(prices)
            }
        
        return None

# ===================== تحلیل تکنیکال پیشرفته ===============================
def get_historical_data(symbol, interval="15m", limit=200):
    """داده‌های تاریخی از Binance"""
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            klines = response.json()
            data_list = []
            for k in klines:
                timestamp = datetime.fromtimestamp(k[0] / 1000)
                open_price = float(k[1])
                high_price = float(k[2])
                low_price = float(k[3])
                close_price = float(k[4])
                volume = float(k[5])
                data_list.append([timestamp, open_price, high_price, low_price, close_price, volume])
            
            df = pd.DataFrame(data_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            return df
    except Exception as e:
        print(f"خطا در دریافت داده‌های تاریخی {symbol}: {e}")
        return None

def calculate_emas(df):
    """محاسبه میانگین‌های متحرک نمایی"""
    try:
        ema_9 = df['close'].ewm(span=9, adjust=False).mean()
        ema_21 = df['close'].ewm(span=21, adjust=False).mean()
        ema_50 = df['close'].ewm(span=50, adjust=False).mean()
        sma_200 = df['close'].rolling(window=200).mean()
        
        return {
            'ema_9': ema_9.iloc[-1] if len(ema_9) > 0 else 0,
            'ema_21': ema_21.iloc[-1] if len(ema_21) > 0 else 0,
            'ema_50': ema_50.iloc[-1] if len(ema_50) > 0 else 0,
            'sma_200': sma_200.iloc[-1] if len(sma_200) > 0 else 0,
            'trend': 'صعودی' if ema_9.iloc[-1] > ema_21.iloc[-1] > ema_50.iloc[-1] else 'نزولی'
        }
    except:
        return {'ema_9': 0, 'ema_21': 0, 'ema_50': 0, 'sma_200': 0, 'trend': 'نامشخص'}

def calculate_bollinger_bands(df, period=20, std=2):
    """محاسبه بولینگر باند"""
    try:
        sma = df['close'].rolling(window=period).mean()
        std_dev = df['close'].rolling(window=period).std()
        
        upper_band = sma + (std_dev * std)
        middle_band = sma
        lower_band = sma - (std_dev * std)
        
        bandwidth = (upper_band - lower_band) / middle_band * 100
        squeeze = bandwidth.iloc[-1] < 10 if len(bandwidth) > 0 else False
        
        return {
            'upper': upper_band.iloc[-1] if len(upper_band) > 0 else 0,
            'middle': middle_band.iloc[-1] if len(middle_band) > 0 else 0,
            'lower': lower_band.iloc[-1] if len(lower_band) > 0 else 0,
            'squeeze': squeeze,
            'bandwidth': bandwidth.iloc[-1] if len(bandwidth) > 0 else 0
        }
    except:
        return {'upper': 0, 'middle': 0, 'lower': 0, 'squeeze': False, 'bandwidth': 0}

def calculate_rsi(df, period=14):
    """محاسبه RSI"""
    try:
        if len(df) < period:
            return 50
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    except:
        return 50

def calculate_macd(df):
    """محاسبه MACD"""
    try:
        if len(df) < 26:
            return {'macd': 0, 'signal': 0, 'histogram': 0, 'trend': 'خنثی'}
        
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        
        macd_value = macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0
        signal_value = signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else 0
        
        trend = 'صعودی' if macd_value > signal_value else 'نزولی'
        
        return {
            'macd': macd_value,
            'signal': signal_value,
            'histogram': histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0,
            'trend': trend
        }
    except:
        return {'macd': 0, 'signal': 0, 'histogram': 0, 'trend': 'خنثی'}

def calculate_advanced_volume(df):
    """تحلیل حجم پیشرفته"""
    try:
        current_volume = df['volume'].iloc[-1]
        avg_volume_20 = df['volume'].rolling(window=20).mean().iloc[-1]
        avg_volume_50 = df['volume'].rolling(window=50).mean().iloc[-1]
        
        if avg_volume_20 > 0:
            volume_ratio_20 = current_volume / avg_volume_20
        else:
            volume_ratio_20 = 1
            
        if avg_volume_50 > 0:
            volume_ratio_50 = current_volume / avg_volume_50
        else:
            volume_ratio_50 = 1
        
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        
        volume_signal = "نرمال"
        if volume_ratio_20 > 3:
            volume_signal = "انفجار حجم"
        elif volume_ratio_20 > 2:
            volume_signal = "حجم بسیار بالا"
        elif volume_ratio_20 > 1.5:
            volume_signal = "حجم بالا"
        elif volume_ratio_20 < 0.5:
            volume_signal = "حجم پایین"
        
        return {
            'volume_ratio': round(volume_ratio_20, 2),
            'current_volume': current_volume,
            'avg_volume_20': avg_volume_20,
            'avg_volume_50': avg_volume_50,
            'volume_signal': volume_signal,
            'vwap': vwap.iloc[-1] if len(vwap) > 0 else 0,
            'vwap_signal': 'بالای VWAP' if df['close'].iloc[-1] > vwap.iloc[-1] else 'زیر VWAP'
        }
    except:
        return {
            'volume_ratio': 1,
            'current_volume': 0,
            'avg_volume_20': 0,
            'avg_volume_50': 0,
            'volume_signal': "نامشخص",
            'vwap': 0,
            'vwap_signal': 'نامشخص'
        }

def detect_candlestick_patterns(df):
    """تشخیص الگوهای کندل‌استیک"""
    patterns = []
    
    try:
        if len(df) < 3:
            return patterns
        
        c1 = df.iloc[-3]
        c2 = df.iloc[-2]
        c3 = df.iloc[-1]
        
        # پین‌بار (Pin Bar)
        c3_body = abs(c3['close'] - c3['open'])
        c3_upper_wick = c3['high'] - max(c3['open'], c3['close'])
        c3_lower_wick = min(c3['open'], c3['close']) - c3['low']
        
        if c3_upper_wick > c3_body * 2 and c3_lower_wick < c3_body * 0.5:
            patterns.append("پین‌بار نزولی")
        elif c3_lower_wick > c3_body * 2 and c3_upper_wick < c3_body * 0.5:
            patterns.append("پین‌بار صعودی")
        
        # انگالفینگ (Engulfing)
        if c2['close'] < c2['open'] and c3['close'] > c3['open']:
            if c3['open'] < c2['close'] and c3['close'] > c2['open']:
                patterns.append("انگالفینگ صعودی")
        elif c2['close'] > c2['open'] and c3['close'] < c3['open']:
            if c3['open'] > c2['close'] and c3['close'] < c2['open']:
                patterns.append("انگالفینگ نزولی")
        
        # همر (Hammer)
        if c3_lower_wick > c3_body * 2 and c3_upper_wick < c3_body * 0.3:
            patterns.append("چکش صعودی")
        
        return patterns
    except:
        return patterns

def calculate_support_resistance(df):
    """محاسبه سطوح حمایت و مقاومت"""
    try:
        recent_df = df.tail(50)
        
        high_points = []
        for i in range(1, len(recent_df)-1):
            if recent_df['high'].iloc[i] > recent_df['high'].iloc[i-1] and \
               recent_df['high'].iloc[i] > recent_df['high'].iloc[i+1]:
                high_points.append(recent_df['high'].iloc[i])
        
        low_points = []
        for i in range(1, len(recent_df)-1):
            if recent_df['low'].iloc[i] < recent_df['low'].iloc[i-1] and \
               recent_df['low'].iloc[i] < recent_df['low'].iloc[i+1]:
                low_points.append(recent_df['low'].iloc[i])
        
        resistance = max(high_points) if high_points else 0
        support = min(low_points) if low_points else 0
        
        current_price = df['close'].iloc[-1]
        
        if resistance > 0 and support > 0:
            price_position = (current_price - support) / (resistance - support) * 100
        else:
            price_position = 50
        
        return {
            'resistance': resistance,
            'support': support,
            'price_position': price_position,
            'signal': 'نزدیک مقاومت' if price_position > 70 else 
                     'نزدیک حمایت' if price_position < 30 else 
                     'وسط رنج'
        }
    except:
        return {
            'resistance': 0,
            'support': 0,
            'price_position': 50,
            'signal': 'نامشخص'
        }

def calculate_dynamic_score(indicators, market_condition="normal"):
    """محاسبه امتیاز پویا"""
    score = 50
    
    if market_condition == "bullish":
        weights = {'rsi': 0.2, 'macd': 0.2, 'volume': 0.25, 'ema': 0.15, 'bb': 0.1, 'sr': 0.1}
    elif market_condition == "bearish":
        weights = {'rsi': 0.25, 'macd': 0.25, 'volume': 0.2, 'ema': 0.1, 'bb': 0.1, 'sr': 0.1}
    else:
        weights = {'rsi': 0.2, 'macd': 0.2, 'volume': 0.2, 'ema': 0.15, 'bb': 0.15, 'sr': 0.1}
    
    # RSI
    rsi = indicators.get('rsi', 50)
    if rsi < 30:
        score += 20 * weights['rsi']
    elif rsi < 40:
        score += 10 * weights['rsi']
    elif rsi > 70:
        score -= 20 * weights['rsi']
    elif rsi > 60:
        score -= 10 * weights['rsi']
    
    # MACD
    macd_trend = indicators.get('macd_trend', 'خنثی')
    if macd_trend == 'صعودی':
        score += 15 * weights['macd']
    elif macd_trend == 'نزولی':
        score -= 15 * weights['macd']
    
    # حجم
    volume_ratio = indicators.get('volume_ratio', 1)
    if volume_ratio > 2:
        score += 20 * weights['volume']
    elif volume_ratio > 1.5:
        score += 10 * weights['volume']
    elif volume_ratio < 0.5:
        score -= 10 * weights['volume']
    
    # EMA
    ema_trend = indicators.get('ema_trend', 'نامشخص')
    if ema_trend == 'صعودی':
        score += 10 * weights['ema']
    elif ema_trend == 'نزولی':
        score -= 10 * weights['ema']
    
    # بولینگر
    bb_squeeze = indicators.get('bb_squeeze', False)
    if bb_squeeze:
        score += 15 * weights['bb']
    
    # حمایت/مقاومت
    sr_signal = indicators.get('sr_signal', 'نامشخص')
    if sr_signal == 'نزدیک حمایت':
        score += 10 * weights['sr']
    elif sr_signal == 'نزدیک مقاومت':
        score -= 10 * weights['sr']
    
    score = max(0, min(100, score))
    
    return round(score, 1)

def analyze_symbol_complete(symbol, connector, timeframe="15m"):
    """تحلیل کامل یک ارز"""
    try:
        price_data = connector.get_best_price_with_volume(symbol)
        if not price_data:
            return None
        
        df = get_historical_data(symbol, interval=timeframe, limit=200)
        if df is None or len(df) < 50:
            return None
        
        rsi = calculate_rsi(df)
        macd_data = calculate_macd(df)
        ema_data = calculate_emas(df)
        bb_data = calculate_bollinger_bands(df)
        volume_data = calculate_advanced_volume(df)
        sr_data = calculate_support_resistance(df)
        candle_patterns = detect_candlestick_patterns(df)
        
        current_price = price_data['price']
        market_condition = "normal"
        if ema_data['trend'] == 'صعودی' and current_price > ema_data['ema_50']:
            market_condition = "bullish"
        elif ema_data['trend'] == 'نزولی' and current_price < ema_data['ema_50']:
            market_condition = "bearish"
        
        indicators = {
            'rsi': rsi,
            'macd_trend': macd_data['trend'],
            'volume_ratio': volume_data['volume_ratio'],
            'ema_trend': ema_data['trend'],
            'bb_squeeze': bb_data['squeeze'],
            'sr_signal': sr_data['signal']
        }
        
        score = calculate_dynamic_score(indicators, market_condition)
        
        if score >= 80:
            signal = "🚀 PUMP ALERT"
            signal_color = "#10b981"
            action = "خرید قوی"
            pump_potential = score
            dump_potential = 100 - score
        elif score >= 70:
            signal = "📈 PUMP WATCH"
            signal_color = "#f59e0b"
            action = "خرید با احتیاط"
            pump_potential = score
            dump_potential = 100 - score
        elif score <= 30:
            signal = "🔴 DUMP ALERT"
            signal_color = "#dc2626"
            action = "فروش قوی"
            pump_potential = score
            dump_potential = 100 - score
        elif score <= 40:
            signal = "📉 DUMP WATCH"
            signal_color = "#ef4444"
            action = "فروش با احتیاط"
            pump_potential = score
            dump_potential = 100 - score
        else:
            signal = "⚪ NEUTRAL"
            signal_color = "#6b7280"
            action = "نگهداری"
            pump_potential = score
            dump_potential = 100 - score
        
        current_price = price_data['price']
        atr = bb_data['bandwidth'] / 100 * current_price if bb_data['bandwidth'] > 0 else current_price * 0.02
        
        if signal in ["🚀 PUMP ALERT", "📈 PUMP WATCH"]:
            entry = current_price
            stop_loss = current_price * 0.97
            stop_loss = min(stop_loss, sr_data['support'] * 0.99) if sr_data['support'] > 0 else stop_loss
            stop_loss = round(stop_loss, 4)
            
            tp1 = current_price * 1.05
            tp2 = current_price * 1.10
            tp3 = current_price * 1.15
            
            if sr_data['resistance'] > 0:
                tp1 = min(tp1, sr_data['resistance'] * 0.98)
                tp2 = min(tp2, sr_data['resistance'] * 1.05)
            
        elif signal in ["🔴 DUMP ALERT", "📉 DUMP WATCH"]:
            entry = current_price
            stop_loss = current_price * 1.03
            stop_loss = max(stop_loss, sr_data['resistance'] * 1.01) if sr_data['resistance'] > 0 else stop_loss
            stop_loss = round(stop_loss, 4)
            
            tp1 = current_price * 0.95
            tp2 = current_price * 0.90
            tp3 = current_price * 0.85
            
            if sr_data['support'] > 0:
                tp1 = max(tp1, sr_data['support'] * 1.02)
                tp2 = max(tp2, sr_data['support'] * 0.95)
        else:
            entry = current_price
            stop_loss = current_price
            tp1 = current_price
            tp2 = current_price
            tp3 = current_price
        
        volume_24h = price_data.get('volume', 0)
        volume_24h_usd = volume_24h * current_price
        
        symbol_clean = symbol.replace('USDT', '')
        if symbol in ["BTCUSDT", "ETHUSDT", "BNBUSDT"]:
            coin_type = "اصلی"
        elif symbol in ["PEPEUSDT", "SHIBUSDT", "FLOKIUSDT", "BONKUSDT", "WIFUSDT"]:
            coin_type = "مم‌کوین"
        else:
            coin_type = "آلت‌کوین"
        
        analysis_summary = []
        if rsi < 35:
            analysis_summary.append("RSI در اشباع فروش")
        if volume_data['volume_ratio'] > 2:
            analysis_summary.append("حجم بسیار بالا")
        if bb_data['squeeze']:
            analysis_summary.append("بولینگر فشرده")
        if "انگالفینگ صعودی" in candle_patterns:
            analysis_summary.append("الگوی انگالفینگ صعودی")
        if sr_data['signal'] == 'نزدیک حمایت':
            analysis_summary.append("نزدیک حمایت کلیدی")
        
        return {
            'symbol': symbol,
            'symbol_clean': symbol_clean,
            'coin_type': coin_type,
            'price': current_price,
            'source': price_data['source'],
            'volume_24h_usd': volume_24h_usd,
            'available_exchanges': price_data['available_exchanges'],
            'exchange_count': price_data['exchange_count'],
            
            'rsi': round(rsi, 1),
            'rsi_signal': "اشباع فروش" if rsi < 30 else 
                         "نزدیک اشباع فروش" if rsi < 40 else 
                         "اشباع خرید" if rsi > 70 else 
                         "نزدیک اشباع خرید" if rsi > 60 else "نرمال",
            
            'macd': round(macd_data['macd'], 4),
            'macd_signal': macd_data['trend'],
            'macd_histogram': round(macd_data['histogram'], 4),
            
            'ema_9': round(ema_data['ema_9'], 4),
            'ema_21': round(ema_data['ema_21'], 4),
            'ema_50': round(ema_data['ema_50'], 4),
            'ema_trend': ema_data['trend'],
            
            'bb_upper': round(bb_data['upper'], 4),
            'bb_middle': round(bb_data['middle'], 4),
            'bb_lower': round(bb_data['lower'], 4),
            'bb_squeeze': bb_data['squeeze'],
            'bb_signal': "فشرده" if bb_data['squeeze'] else "عادی",
            
            'volume_ratio': volume_data['volume_ratio'],
            'volume_signal': volume_data['volume_signal'],
            'vwap': round(volume_data['vwap'], 4),
            'vwap_signal': volume_data['vwap_signal'],
            
            'support': round(sr_data['support'], 4),
            'resistance': round(sr_data['resistance'], 4),
            'sr_signal': sr_data['signal'],
            
            'candle_patterns': ", ".join(candle_patterns) if candle_patterns else "ندارد",
            
            'score': score,
            'signal': signal,
            'signal_color': signal_color,
            'action': action,
            'pump_potential': pump_potential,
            'dump_potential': dump_potential,
            
            'entry': round(entry, 4),
            'stop_loss': stop_loss,
            'tp1': round(tp1, 4),
            'tp2': round(tp2, 4),
            'tp3': round(tp3, 4),
            
            'analysis_summary': analysis_summary,
            'market_condition': market_condition,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        print(f"⚠️ خطا در تحلیل {symbol}: {str(e)}")
        return None

# ===================== UI اصلی ===============================
def main():
    if 'connector' not in st.session_state:
        st.session_state.connector = TripleExchangeConnector()
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    
    if 'refresh_count' not in st.session_state:
        st.session_state.refresh_count = 0
    
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 120
    
    # هدر اصلی
    st.title("🚀 Mojtaba_D Trader Pro ULTIMATE - PUMP & DUMP EDITION")
    st.markdown("**نسخه ویژه ۱۵ ارز پامپ‌خیز و دامپ‌خیز**")
    
    # آمار
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📡 صرافی‌ها", "۳")
    
    with col2:
        st.metric("🚀 ارزها", "۱۵")
    
    with col3:
        st.metric("📊 اندیکاتورها", "۱۰+")
    
    with col4:
        st.metric("🔄 تحلیل شماره", st.session_state.refresh_count)
    
    with col5:
        st.metric("⏰ آخرین بروزرسانی", datetime.now().strftime('%H:%M'))
    
    st.markdown("---")
    
    # نوار کناری
    with st.sidebar:
        st.title("⚙️ کنترل پنل PUMP & DUMP")
        
        # انتخاب تایم‌فرم
        st.markdown("### 📈 تایم‌فرم تحلیل")
        timeframe = st.selectbox(
            "بازه زمانی",
            ["15m", "1h", "4h", "1d"],
            index=0,
            key="timeframe"
        )
        
        st.markdown(f"**تایم‌فرم انتخابی:** {timeframe}")
        
        # دکمه تحلیل
        if st.button("🚀 تحلیل ۱۵ ارز پامپ‌خیز", type="primary", use_container_width=True):
            with st.spinner(f"در حال تحلیل ۱۵ ارز پامپ‌خیز..."):
                results = []
                progress_bar = st.progress(0)
                
                for idx, symbol in enumerate(SYMBOLS):
                    analysis = analyze_symbol_complete(symbol, st.session_state.connector, timeframe)
                    if analysis:
                        results.append(analysis)
                    
                    progress = (idx + 1) / len(SYMBOLS)
                    progress_bar.progress(progress)
                    time.sleep(0.1)
                
                st.session_state.analysis_results = results
                st.session_state.refresh_count += 1
                
                st.success(f"✅ {len(results)} ارز تحلیل شد")
                st.balloons()
        
        st.markdown("---")
        
        # سیستم رفرش خودکار
        st.markdown("### 🔄 رفرش خودکار")
        auto_refresh = st.checkbox("فعال‌سازی رفرش خودکار", value=False, key="auto_refresh_check")
        
        if auto_refresh:
            interval = st.select_slider(
                "بازه رفرش",
                options=["۳۰ ثانیه", "۱ دقیقه", "۲ دقیقه", "۵ دقیقه", "۱۰ دقیقه"],
                value="۲ دقیقه",
                key="refresh_interval_slider"
            )
            
            intervals_map = {
                "۳۰ ثانیه": 30,
                "۱ دقیقه": 60,
                "۲ دقیقه": 120,
                "۵ دقیقه": 300,
                "۱۰ دقیقه": 600
            }
            
            st.session_state.auto_refresh = True
            st.session_state.refresh_interval = intervals_map[interval]
            
            if st.button("🔄 رفرش الان", type="secondary"):
                st.rerun()
        else:
            st.session_state.auto_refresh = False
        
        st.markdown("---")
        
        # فیلترها
        st.markdown("### 🔍 فیلترهای پیشرفته")
        
        min_score = st.slider("حداقل امتیاز", 0, 100, 70, key="min_score")
        
        signal_type = st.selectbox(
            "نوع سیگنال",
            ["همه", "🚀 PUMP ALERT", "📈 PUMP WATCH", "📉 DUMP WATCH", "🔴 DUMP ALERT"],
            key="signal_type"
        )
        
        coin_type = st.selectbox(
            "نوع ارز",
            ["همه", "مم‌کوین", "آلت‌کوین", "اصلی"],
            key="coin_type"
        )
        
        min_volume = st.number_input(
            "حداقل حجم ۲۴h (میلیون دلار)",
            min_value=0,
            max_value=1000,
            value=10,
            key="min_volume"
        ) * 1000000
        
        min_exchanges = st.slider("حداقل صرافی", 1, 3, 2, key="min_exchanges")
        
        st.markdown("---")
        
        # تنظیمات آلارم
        st.markdown("### 🔔 تنظیمات آلارم PUMP")
        
        enable_alert = st.checkbox("فعال‌سازی آلارم صوتی", value=True)
        
        alert_score = st.slider("آلارم برای امتیاز بالای", 0, 100, 80, key="alert_score")
        
        if enable_alert:
            st.info("🔔 آلارم برای سیگنال‌های قوی فعال شد")
        
        st.markdown("---")
        
        # آمار سیستم
        st.markdown("### 📊 آمار سیستم")
        st.write(f"📅 {datetime.now().strftime('%Y-%m-%d')}")
        st.write(f"🕒 {datetime.now().strftime('%H:%M:%S')}")
        st.write(f"🔄 {st.session_state.refresh_count} تحلیل")
        st.write(f"🚀 {len(SYMBOLS)} ارز پامپ‌خیز")
    
    # سیستم رفرش خودکار
    if st.session_state.auto_refresh:
        time.sleep(st.session_state.refresh_interval)
        st.rerun()
    
    # تب‌های اصلی
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 ۵ سیگنال برتر پامپ", 
        "📉 هشدارهای دامپ", 
        "💰 همه ارزها",
        "📊 تحلیل گرافیکی"
    ])
    
    with tab1:
        display_top_pump_signals(min_score, signal_type, coin_type, min_volume, min_exchanges)
    
    with tab2:
        display_dump_alerts(min_score, min_volume, min_exchanges)
    
    with tab3:
        display_all_coins(min_score, signal_type, coin_type, min_volume, min_exchanges)
    
    with tab4:
        display_technical_analysis()
    
    # فوتر
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center;">
        <p>🤖 <b>Mojtaba_D Trader Pro - PUMP & DUMP EDITION</b> | 
        📡 <b>۳ صرافی</b> | 🚀 <b>۱۵ ارز پامپ‌خیز</b> | 
        🕒 <b>{datetime.now().strftime('%H:%M:%S')}</b></p>
        <p style="font-size: 12px; color: #666;">
        ⚠️ این ابزار برای کمک به تصمیم‌گیری است و مسئولیتی در قبال سود/ضرر ندارد.
        نوسان‌گیری پرریسک است، مدیریت سرمایه فراموش نشود.
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_top_pump_signals(min_score, signal_type, coin_type, min_volume, min_exchanges):
    """نمایش ۵ سیگنال برتر پامپ"""
    st.markdown("## 🚀 ۵ سیگنال برتر پامپ")
    
    if not st.session_state.analysis_results:
        st.info("⏳ هنوز تحلیلی انجام نشده. روی دکمه 'تحلیل ۱۵ ارز پامپ‌خیز' کلیک کنید.")
        return
    
    filtered_coins = []
    for coin in st.session_state.analysis_results:
        if coin['score'] >= min_score:
            if signal_type == "همه" or coin['signal'] == signal_type:
                if coin_type == "همه" or coin['coin_type'] == coin_type:
                    if coin['volume_24h_usd'] >= min_volume:
                        if coin['exchange_count'] >= min_exchanges:
                            if coin['signal'] in ["🚀 PUMP ALERT", "📈 PUMP WATCH"]:
                                filtered_coins.append(coin)
    
    if filtered_coins:
        filtered_coins.sort(key=lambda x: x['score'], reverse=True)
        top_5 = filtered_coins[:5]
        
        st.markdown("### 🏆 بهترین فرصت‌های پامپ")
        cols = st.columns(5)
        
        for idx, coin in enumerate(top_5):
            with cols[idx]:
                if coin['coin_type'] == "مم‌کوین":
                    border_color = "#8b5cf6"
                    bg_gradient = "linear-gradient(135deg, #8b5cf620, #8b5cf640)"
                elif coin['coin_type'] == "اصلی":
                    border_color = "#3b82f6"
                    bg_gradient = "linear-gradient(135deg, #3b82f620, #3b82f640)"
                else:
                    border_color = "#10b981"
                    bg_gradient = "linear-gradient(135deg, #10b98120, #10b98140)"
                
                st.markdown(f"""
                <div style="
                    background: {bg_gradient};
                    border: 2px solid {border_color};
                    border-radius: 10px;
                    padding: 15px;
                    margin: 10px 0;
                    text-align: center;
                ">
                    <h3 style="margin: 0;">{coin['symbol_clean']}</h3>
                    <p style="font-size: 20px; margin: 5px 0;">
                        ${coin['price']:,.4f}
                    </p>
                    <div style="display: flex; justify-content: center; align-items: center; gap: 5px; margin: 10px 0;">
                        <span style="color: #10b981; font-weight: bold; font-size: 18px;">
                            🎯 {coin['score']}/100
                        </span>
                        <span style="background-color: {border_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px;">
                            {coin['coin_type']}
                        </span>
                    </div>
                    <p style="margin: 5px 0; font-weight: bold;">{coin['signal']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # جزئیات بهترین سیگنال
        if top_5:
            best_coin = top_5[0]
            with st.expander(f"**🎯 بهترین سیگنال: {best_coin['symbol_clean']}**", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("💰 قیمت", f"${best_coin['price']:,.4f}")
                    st.caption(f"منبع: {best_coin['source']}")
                
                with col2:
                    st.metric("🎯 امتیاز", f"{best_coin['score']}/100")
                    st.caption(best_coin['signal'])
                
                with col3:
                    st.metric("📊 RSI", f"{best_coin['rsi']:.1f}")
                    st.caption(best_coin['rsi_signal'])
                
                with col4:
                    st.metric("📈 حجم", f"{best_coin['volume_ratio']:.1f}x")
                    st.caption(best_coin['volume_signal'])
                
                # پیشنهاد معامله
                st.markdown("#### 🎯 پیشنهاد معامله پامپ:")
                
                profit_1 = ((best_coin['tp1'] - best_coin['entry']) / best_coin['entry'] * 100)
                profit_2 = ((best_coin['tp2'] - best_coin['entry']) / best_coin['entry'] * 100)
                profit_3 = ((best_coin['tp3'] - best_coin['entry']) / best_coin['entry'] * 100)
                loss = ((best_coin['entry'] - best_coin['stop_loss']) / best_coin['entry'] * 100)
                
                col5, col6, col7, col8 = st.columns(4)
                
                with col5:
                    st.metric("🎯 ورود", f"${best_coin['entry']:,.4f}")
                
                with col6:
                    st.metric("🛑 حد ضرر", f"${best_coin['stop_loss']:,.4f}", delta=f"-{loss:.1f}%")
                
                with col7:
                    st.metric("✅ هدف ۱", f"${best_coin['tp1']:,.4f}", delta=f"+{profit_1:.1f}%")
                
                with col8:
                    st.metric("🎯 هدف ۲", f"${best_coin['tp2']:,.4f}", delta=f"+{profit_2:.1f}%")
                
                st.metric("🚀 هدف ۳", f"${best_coin['tp3']:,.4f}", delta=f"+{profit_3:.1f}%")
    else:
        st.warning("⚠️ در حال حاضر هیچ سیگنال پامپ قوی مطابق فیلترها شناسایی نشد.")

def display_dump_alerts(min_score, min_volume, min_exchanges):
    """نمایش هشدارهای دامپ"""
    st.markdown("## 📉 هشدارهای ریسک دامپ")
    
    if not st.session_state.analysis_results:
        st.info("⏳ هنوز تحلیلی انجام نشده.")
        return
    
    dump_coins = []
    for coin in st.session_state.analysis_results:
        if coin['dump_potential'] >= 70:
            if coin['volume_24h_usd'] >= min_volume:
                if coin['exchange_count'] >= min_exchanges:
                    if coin['signal'] in ["🔴 DUMP ALERT", "📉 DUMP WATCH"]:
                        dump_coins.append(coin)
    
    if dump_coins:
        dump_coins.sort(key=lambda x: x['dump_potential'], reverse=True)
        
        st.markdown(f"### ⚠️ {len(dump_coins)} ارز با ریسک دامپ بالا")
        
        for coin in dump_coins[:3]:
            st.error(f"**{coin['symbol_clean']}** - ریسک: {coin['dump_potential']}/100 - سیگنال: {coin['signal']}")
            st.write(f"قیمت: ${coin['price']:,.4f} | RSI: {coin['rsi']:.1f} | حجم: {coin['volume_ratio']:.1f}x")
    else:
        st.success("✅ وضعیت خوب - هیچ ارز با ریسک دامپ بالا شناسایی نشد.")

def display_all_coins(min_score, signal_type, coin_type, min_volume, min_exchanges):
    """نمایش همه ارزها"""
    st.markdown("## 💰 همه ارزهای پامپ‌خیز")
    
    if not st.session_state.analysis_results:
        st.info("⏳ هنوز تحلیلی انجام نشده.")
        return
    
    filtered_coins = []
    for coin in st.session_state.analysis_results:
        if coin['score'] >= min_score:
            if signal_type == "همه" or coin['signal'] == signal_type:
                if coin_type == "همه" or coin['coin_type'] == coin_type:
                    if coin['volume_24h_usd'] >= min_volume:
                        if coin['exchange_count'] >= min_exchanges:
                            filtered_coins.append(coin)
    
    if filtered_coins:
        table_data = []
        for coin in filtered_coins:
            table_data.append({
                'نماد': coin['symbol_clean'],
                'نوع': coin['coin_type'],
                'سیگنال': coin['signal'],
                'قیمت': f"${coin['price']:,.4f}",
                'امتیاز': coin['score'],
                'RSI': f"{coin['rsi']:.1f}",
                'حجم': f"{coin['volume_ratio']:.1f}x",
                'صرافی': coin['exchange_count']
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ هیچ ارزی مطابق فیلترها یافت نشد.")

def display_technical_analysis():
    """نمایش تحلیل تکنیکال"""
    st.markdown("## 📊 تحلیل تکنیکال")
    
    if not st.session_state.analysis_results:
        st.info("⏳ ابتدا تحلیل را اجرا کنید.")
        return
    
    st.info("این بخش نیاز به Plotly دارد. در نسخه Termux ممکن است نمایش داده نشود.")
    st.write("برای تحلیل گرافیکی کامل، نسخه دسکتاپ را اجرا کنید.")

if __name__ == "__main__":
    main()
