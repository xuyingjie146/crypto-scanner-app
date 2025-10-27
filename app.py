from flask import Flask, render_template, request, jsonify, send_file
import sys
import os
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from scipy.signal import argrelextrema
from scipy.stats import linregress
import warnings
import io
import base64
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

app = Flask(__name__)

class WebPatternScanner:
    def __init__(self):
        self.base_url = "https://api.gateio.ws/api/v4"
        self.timeframes = ["15m", "1h", "4h", "1d"]
        
    def get_top_symbols(self, limit=20):
        """获取交易量前20的币种"""
        try:
            url = f"{self.base_url}/spot/tickers"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                tickers_data = response.json()
                usdt_pairs = []
                
                for ticker in tickers_data:
                    currency_pair = ticker.get('currency_pair', '')
                    if currency_pair.endswith('_USDT'):
                        quote_volume = float(ticker.get('quote_volume', 0))
                        usdt_pairs.append({
                            'symbol': currency_pair,
                            'quote_volume': quote_volume
                        })
                
                usdt_pairs.sort(key=lambda x: x['quote_volume'], reverse=True)
                return [item['symbol'] for item in usdt_pairs[:limit]]
            else:
                return self.get_backup_symbols(limit)
                
        except Exception:
            return self.get_backup_symbols(limit)
    
    def get_backup_symbols(self, limit=20):
        """备用币种列表"""
        return [
            "BTC_USDT", "ETH_USDT", "BNB_USDT", "SOL_USDT", "XRP_USDT",
            "ADA_USDT", "AVAX_USDT", "DOGE_USDT", "DOT_USDT", "LINK_USDT",
            "MATIC_USDT", "LTC_USDT", "ATOM_USDT", "ETC_USDT", "XLM_USDT"
        ][:limit]

    def get_candle_data(self, symbol="BTC_USDT", interval="15m", limit=200):
        """获取K线数据"""
        try:
            url = f"{self.base_url}/spot/candlesticks"
            params = {
                'currency_pair': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self.process_candle_data(data)
            else:
                return self.generate_simulated_data(symbol, interval, limit)
                
        except Exception:
            return self.generate_simulated_data(symbol, interval, limit)

    def process_candle_data(self, data):
        """处理K线数据"""
        try:
            processed_data = []
            for candle in data:
                if len(candle) >= 6:
                    timestamp = int(candle[0])
                    close = float(candle[2])
                    high = float(candle[3])
                    low = float(candle[4])
                    open_price = float(candle[5])
                    
                    processed_data.append({
                        'timestamp': pd.to_datetime(timestamp * 1000),
                        'Open': open_price,
                        'High': high,
                        'Low': low,
                        'Close': close
                    })
            
            if not processed_data:
                return None
                
            df = pd.DataFrame(processed_data)
            df.set_index('timestamp', inplace=True)
            return df.sort_index()
            
        except Exception:
            return None

    def generate_simulated_data(self, symbol, interval, limit=200):
        """生成模拟数据"""
        real_prices = {
            "BTC_USDT": 45000, "ETH_USDT": 2500, "BNB_USDT": 320,
            "SOL_USDT": 110, "XRP_USDT": 0.62, "ADA_USDT": 0.48
        }
        
        base_price = real_prices.get(symbol, 10)
        
        # 根据时间框架设置波动率
        volatility = {
            "15m": 0.003, "1h": 0.008, "4h": 0.015, "1d": 0.025
        }.get(interval, 0.01)
        
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='H')
        np.random.seed(hash(symbol) % 10000)
        
        prices = [base_price]
        for i in range(1, limit):
            change = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            new_price = max(new_price, base_price * 0.1)
            prices.append(new_price)
        
        df_data = []
        for i, date in enumerate(dates):
            close_price = prices[i]
            open_price = prices[i-1] if i > 0 else close_price * 0.99
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, volatility/2)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, volatility/2)))
            
            df_data.append({
                'timestamp': date,
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        return df

    def find_swing_points(self, df, window=5):
        """找到摆动点"""
        if len(df) < window * 2:
            return [], []
        
        highs = df['High'].values
        lows = df['Low'].values
        
        high_indices = argrelextrema(highs, np.greater, order=window)[0]
        low_indices = argrelextrema(lows, np.less, order=window)[0]
        
        swing_highs = []
        swing_lows = []
        
        for idx in high_indices[-2:]:
            if idx < len(highs):
                swing_highs.append((idx, highs[idx]))
        
        for idx in low_indices[-2:]:
            if idx < len(lows):
                swing_lows.append((idx, lows[idx]))
        
        return swing_highs, swing_lows

    def detect_patterns(self, df):
        """检测形态"""
        if df is None or len(df) < 100:
            return "无足够数据", 0, [], []
        
        swing_highs, swing_lows = self.find_swing_points(df)
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "摆动点不足", 0, swing_highs, swing_lows
        
        dates_num = mdates.date2num(df.index.to_pydatetime())
        
        # 简化的形态检测逻辑
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # 计算趋势
            high_slope = (swing_highs[-1][1] - swing_highs[-2][1]) / (swing_highs[-1][0] - swing_highs[-2][0])
            low_slope = (swing_lows[-1][1] - swing_lows[-2][1]) / (swing_lows[-1][0] - swing_lows[-2][0])
            
            if high_slope < -0.001 and low_slope > 0.001:
                return "对称三角形", 85, swing_highs, swing_lows
            elif abs(high_slope) < 0.0005 and low_slope > 0.001:
                return "上升三角形", 80, swing_highs, swing_lows
            elif high_slope < -0.001 and abs(low_slope) < 0.0005:
                return "下降三角形", 80, swing_highs, swing_lows
            elif abs(high_slope - low_slope) < 0.0002:
                if high_slope > 0.001:
                    return "上升通道", 75, swing_highs, swing_lows
                else:
                    return "下降通道", 75, swing_highs, swing_lows
        
        return "未发现明显形态", 0, swing_highs, swing_lows

    def create_chart_image(self, df, symbol, interval, pattern_type, pattern_score, swing_highs, swing_lows):
        """创建图表并返回base64图片"""
        try:
            # 创建图表
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 绘制K线
            dates = df.index
            dates_num = mdates.date2num(dates.to_pydatetime())
            
            for i in range(len(dates)):
                date_num = dates_num[i]
                open_val = df['Open'].iloc[i]
                high_val = df['High'].iloc[i]
                low_val = df['Low'].iloc[i]
                close_val = df['Close'].iloc[i]
                
                color = 'green' if close_val >= open_val else 'red'
                
                # 绘制影线
                ax.plot([date_num, date_num], [low_val, high_val], color='black', linewidth=0.8, alpha=0.7)
                
                # 绘制实体
                body_bottom = min(open_val, close_val)
                body_top = max(open_val, close_val)
                body_height = body_top - body_bottom
                
                if body_height > 0:
                    width = (dates_num[-1] - dates_num[0]) / len(dates_num) * 0.7
                    rect = plt.Rectangle((date_num - width/2, body_bottom), width, body_height, 
                                       facecolor=color, alpha=0.7, edgecolor='black')
                    ax.add_patch(rect)
            
            # 标注摆动点
            for idx, price in swing_highs:
                if idx < len(dates_num):
                    ax.plot(dates_num[idx], price, 'v', color='red', markersize=8, alpha=0.8, label='High' if idx == swing_highs[0][0] else "")
            
            for idx, price in swing_lows:
                if idx < len(dates_num):
                    ax.plot(dates_num[idx], price, '^', color='blue', markersize=8, alpha=0.8, label='Low' if idx == swing_lows[0][0] else "")
            
            # 设置标题
            title = f"{symbol} - {interval}\n{pattern_type} (Score: {pattern_score}%)"
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel('Price (USDT)', fontsize=12)
            
            # 设置日期格式
            date_format = mdates.DateFormatter('%m-%d %H:%M')
            ax.xaxis.set_major_formatter(date_format)
            plt.xticks(rotation=45)
            
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            
            # 转换为base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
            
            return image_base64
            
        except Exception as e:
            print(f"图表创建失败: {e}")
            return None

# 创建扫描器实例
scanner = WebPatternScanner()

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/get_symbols')
def get_symbols():
    """获取币种列表"""
    symbols = scanner.get_top_symbols(20)
    return jsonify({'symbols': symbols})

@app.route('/scan', methods=['POST'])
def scan():
    """执行扫描"""
    try:
        data = request.json
        symbol = data.get('symbol', 'BTC_USDT')
        interval = data.get('interval', '1h')
        kline_count = int(data.get('kline_count', 200))
        
        print(f"开始扫描: {symbol} {interval}")
        
        # 获取数据
        df = scanner.get_candle_data(symbol, interval, kline_count)
        
        # 检测形态
        pattern_type, pattern_score, swing_highs, swing_lows = scanner.detect_patterns(df)
        
        # 生成图表
        chart_image = None
        if pattern_score > 0:
            chart_image = scanner.create_chart_image(df, symbol, interval, pattern_type, pattern_score, swing_highs, swing_lows)
        
        result = {
            'success': True,
            'symbol': symbol,
            'interval': interval,
            'pattern_type': pattern_type,
            'pattern_score': pattern_score,
            'current_price': df['Close'].iloc[-1] if df is not None else 0,
            'swing_highs_count': len(swing_highs),
            'swing_lows_count': len(swing_lows),
            'chart_image': chart_image,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)