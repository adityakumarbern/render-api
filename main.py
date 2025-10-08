import yfinance as yf
import requests
import pandas as pd
import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")

NEWS_API_URL = "https://newsapi.org/v2/everything"

if GOOGLE_API_KEY and GOOGLE_API_KEY != "YOUR_GOOGLE_API_KEY":
    genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI(
    title="Equity Research API",
    description="Provides endpoints for stock history, news, volatility, and AI-powered analysis.",
    version="2.0",
)

class StockRequest(BaseModel):
    tickers: List[str]
    period: Optional[str] = "1y"

class VolatilityRequest(BaseModel):
    tickers: List[str]
    window: Optional[int] = 30
    period: Optional[str] = "1y"

class NewsRequest(BaseModel):
    query: str
    limit: Optional[int] = 10

class Article(BaseModel):
    title: str
    source: str
    published_at: str
    url: str

class NewsResponse(BaseModel):
    query: str
    articles: List[Article]

class Event(BaseModel):
    headline: str
    source: str
    url: Optional[str] = None

class ChartDataPoint(BaseModel):
    date: str
    close_price: float
    event: Optional[Event] = None

class AnalysisChartResponse(BaseModel):
    ticker: str
    chart_data: List[ChartDataPoint]

class SummaryResponse(BaseModel):
    summary: str

# --- START: New Models for Dashboard Endpoint ---
class Metric(BaseModel):
    value: str
    percent_change: float

class DashboardTopResponse(BaseModel):
    market_cap: Metric
    eps: Metric
    revenue: Metric
    daily_percent_move: float
# --- END: New Models for Dashboard Endpoint ---


KEY_EVENTS_DATA = [
    {"date": "2024-10-17", "source": "Bloomberg.com", "headline": "Otsuka Is Said to Weigh Sale of Stake in Medical Device Maker MicroPort Scientific", "url": "https://www.bloomberg.com/news/articles/2024-10-17/otsuka-weighs-sale-of-stake-in-medical-device-maker-microport-scientific"},
    {"date": "2025-03-31", "source": "simplywall.st", "headline": "MicroPort Scientific Full Year 2024 Earnings: EPS Beats Expectations, Revenues Lag", "url": "https://simplywall.st/stocks/hk/healthcare/hkg-853/microport-scientific-shares/news/microport-scientific-full-year-2024-earnings-eps-beats-expec"},
    {"date": "2025-09-03", "source": "Yahoo Finance", "headline": "MicroPort Scientific Corp (MCRPF) (H1 2025) Earnings Call Highlights: Navigating Challenges", "url": "https://finance.yahoo.com/news/microport-scientific-corp-mcrpf-h1-070046770.html"},
]

@app.get("/")
def read_root():
    return {"status": "Equity Research API is running."}

@app.post("/stock/")
async def get_stock_data(request: StockRequest):
    try:
        hist_data = yf.download(tickers=request.tickers, period=request.period, group_by='ticker')
        if hist_data.empty: raise HTTPException(status_code=404, detail=f"Could not retrieve data for tickers: {request.tickers}")
        response_data = {}
        is_single_ticker = len(request.tickers) == 1
        for ticker in request.tickers:
            ticker_df = hist_data.dropna() if is_single_ticker else hist_data[ticker].dropna()
            if not ticker_df.empty:
                ticker_df.reset_index(inplace=True)
                ticker_df['Date'] = ticker_df['Date'].dt.strftime('%Y-%m-%d')
                response_data[ticker] = ticker_df.to_dict(orient="records")
            else: response_data[ticker] = []
        return response_data
    except Exception as e: raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/volatility/")
async def get_volatility_data(request: VolatilityRequest):
    try:
        hist_data = yf.download(tickers=request.tickers, period=request.period)
        if hist_data.empty: raise HTTPException(status_code=404, detail=f"Could not retrieve data for tickers: {request.tickers}")
        is_single_ticker = len(request.tickers) == 1
        close_prices = hist_data['Close'] if not is_single_ticker else hist_data[['Close']]
        if is_single_ticker: close_prices.columns = [request.tickers[0]]
        daily_returns = close_prices.pct_change()
        rolling_volatility = daily_returns.rolling(window=request.window).std() * (252**0.5)
        response_data = {}
        for ticker in request.tickers:
            vol_series = rolling_volatility[ticker].dropna()
            data_list = [{"Date": date.strftime('%Y-%m-%d'), "Volatility": f"{vol:.4f}"} for date, vol in vol_series.items()]
            response_data[ticker] = data_list
        return response_data
    except Exception as e: raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/volatility/snapshot/")
async def get_volatility_snapshot():
    try:
        tickers = ["0853.HK", "^HSI", "MDT"]
        hist_data = yf.download(tickers=tickers, period="1y")
        if hist_data.empty: raise HTTPException(status_code=404, detail="Could not retrieve snapshot data.")
        close_prices = hist_data['Close']
        daily_returns = close_prices.pct_change()
        response = {}
        for window in [30, 60, 90]:
            rolling_vol = daily_returns.rolling(window=window).std() * (252**0.5)
            latest_vol_target = rolling_vol["0853.HK"].iloc[-1]
            latest_vol_benchmark = rolling_vol["^HSI"].iloc[-1]
            latest_vol_peer = rolling_vol["MDT"].iloc[-1]
            avg_comparison_vol = (latest_vol_benchmark + latest_vol_peer) / 2
            comparison = {
                "vs_benchmark": round(latest_vol_target / latest_vol_benchmark, 2),
                "vs_peer": round(latest_vol_target / latest_vol_peer, 2),
                "vs_average": round(latest_vol_target / avg_comparison_vol, 2)
            }
            response[f"{window}-day"] = {
                "target": {"ticker": "0853.HK", "volatility": round(latest_vol_target, 4)},
                "benchmark": {"ticker": "^HSI", "volatility": round(latest_vol_benchmark, 4)},
                "peer": {"ticker": "MDT", "volatility": round(latest_vol_peer, 4)},
                "comparison": comparison
            }
        return response
    except Exception as e: raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/news/", response_model=NewsResponse)
async def get_news_articles(request: NewsRequest):
    params = {"q": request.query, "apiKey": NEWS_API_KEY, "language": "en", "sortBy": "publishedAt", "pageSize": request.limit}
    try:
        response = requests.get(NEWS_API_URL, params=params)
        response.raise_for_status()
        articles_data = response.json().get("articles", [])
        formatted_articles = [Article(title=a.get("title", "No Title"), source=a.get("source", {}).get("name", "Unknown"), published_at=a.get("publishedAt", ""), url=a.get("url", "")) for a in articles_data]
        return NewsResponse(query=request.query, articles=formatted_articles)
    except requests.exceptions.RequestException as e: raise HTTPException(status_code=503, detail=f"Error communicating with NewsAPI: {e}")

@app.get("/analysis/chart", response_model=AnalysisChartResponse)
async def get_analysis_chart(period: str = "1y"):
    try:
        target_ticker, benchmark_ticker = "0853.HK", "^HSI"
        hist_data = yf.download(tickers=[target_ticker, benchmark_ticker], period=period)
        if hist_data.empty: raise HTTPException(status_code=404, detail="Could not retrieve analysis data.")
        analysis_df = pd.DataFrame({'target_close': hist_data[('Close', target_ticker)], 'benchmark_close': hist_data[('Close', benchmark_ticker)]}).dropna()
        analysis_df['target_return'] = analysis_df['target_close'].pct_change()
        analysis_df['benchmark_return'] = analysis_df['benchmark_close'].pct_change()
        analysis_df['abnormal_return'] = analysis_df['target_return'] - analysis_df['benchmark_return']
        std_dev = analysis_df['abnormal_return'].std()
        spike_threshold = 1.75 * std_dev
        events_df = pd.DataFrame(KEY_EVENTS_DATA); events_df['date'] = pd.to_datetime(events_df['date']); events_df.set_index('date', inplace=True)
        chart_data_points = {}
        spike_dates = analysis_df[analysis_df['abnormal_return'].abs() > spike_threshold].index
        for date in spike_dates:
            search_window = pd.date_range(start=date - pd.Timedelta(days=2), end=date)
            matched_events = events_df.loc[events_df.index.isin(search_window)]
            if not matched_events.empty:
                event_info = matched_events.iloc[0]
                chart_data_points[date.strftime('%Y-%m-%d')] = Event(headline=event_info['headline'], source=event_info['source'], url=event_info['url'])
        for event_date, event_info in events_df.iterrows():
            chart_data_points[event_date.strftime('%Y-%m-%d')] = Event(headline=event_info['headline'], source=event_info['source'], url=event_info['url'])
        final_chart_data = [ChartDataPoint(date=index.strftime('%Y-%m-%d'), close_price=row['target_close'], event=chart_data_points.get(index.strftime('%Y-%m-%d'))) for index, row in analysis_df.iterrows()]
        return AnalysisChartResponse(ticker=target_ticker, chart_data=final_chart_data)
    except Exception as e: raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/analysis/summary", response_model=SummaryResponse)
async def get_ai_summary():
    try:
        snapshot_data = await get_volatility_snapshot()
        vol_30d = snapshot_data['30-day']['target']['volatility']
        comparison_factor = snapshot_data['30-day']['comparison']['vs_average']
        
        price_data = yf.download(tickers=["0853.HK"], period="2d")
        price_change_percent = price_data['Close'].pct_change().iloc[-1] * 100
        
        latest_event = sorted(KEY_EVENTS_DATA, key=lambda x: x['date'], reverse=True)[0]

        vol_30d_float = float(vol_30d)
        comparison_factor_float = float(comparison_factor)
        price_change_float = float(price_change_percent)
        
        prompt = f'''
        You are a financial analyst providing a brief, objective summary for a dashboard. Based on the following data for MicroPort Scientific (0853.HK), generate a 2-3 sentence professional summary. Do not give financial advice or use speculative language.

        - Current 30-Day Volatility: {vol_30d_float:.2%}
        - Volatility vs Market/Peer Average: {comparison_factor_float:.2f}x higher
        - Last 24h Price Change: {price_change_float:.2f}%
        - Latest Major Event ({latest_event['date']}): {latest_event['headline']}

        Generate the summary now.
        '''

        if GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY":
            raise HTTPException(status_code=500, detail="Google AI API key not configured.")
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        
        return SummaryResponse(summary=response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while generating the AI summary: {str(e)}")

@app.get("/dashboard/top", response_model=DashboardTopResponse)
async def get_top_dashboard_data():

    try:
        ticker_symbol = "0853.HK"
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        hist = yf.download(ticker_symbol, period="2d", progress=False)
        if hist.empty or len(hist) < 2:
            raise HTTPException(status_code=404, detail="Not enough historical data for daily move calculation.")
        
       
        daily_move_series = hist['Close'].pct_change().tail(1)
        daily_move = daily_move_series.item() if not daily_move_series.empty else None

        def to_percentage_float(value: Optional[float]) -> float:
            if value is None or pd.isna(value):
                return 0.0
            return float(value * 100)


        def format_large_number(num: Optional[float]) -> str:
            if num is None: return "N/A"
            if abs(num) >= 1_000_000_000:
                return f"${num / 1_000_000_000:.2f}B"
            if abs(num) >= 1_000_000:
                return f"${num / 1_000_000:.2f}M"
            return f"${num:,.2f}"

        response_data = {
            "market_cap": {
                "value": format_large_number(info.get('marketCap')),
                "percent_change": to_percentage_float(daily_move)
            },
            "eps": {
                "value": f"${info.get('trailingEps', 0):.2f}",
                "percent_change": to_percentage_float(info.get('earningsQuarterlyGrowth'))
            },
            "revenue": {
                "value": format_large_number(info.get('totalRevenue')),
                "percent_change": to_percentage_float(info.get('revenueGrowth'))
            },
            "daily_percent_move": to_percentage_float(daily_move)
        }
        
        return response_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching dashboard data: {str(e)}")
