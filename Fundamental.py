import yfinance as yf
import requests
from bs4 import BeautifulSoup

# Step 1: Fetch Financial Data
def fetch_financial_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    
    # Get financial data (if available)
    try:
        cash_flow = ticker.cashflow
        net_income = ticker.financials.loc["Net Income"]
    except:
        cash_flow = None
        net_income = None
    
    # Get key metrics
    pe_ratio = ticker.info.get("trailingPE", None)
    revenue_growth = ticker.info.get("revenueGrowth", None)
    market_cap = ticker.info.get("marketCap", None)
    
    return {
        "cash_flow": cash_flow,
        "net_income": net_income,
        "pe_ratio": pe_ratio,
        "revenue_growth": revenue_growth,
        "market_cap": market_cap
    }

# Step 2: Fetch Leadership Data from Yahoo Finance
def fetch_leadership_data(ticker_symbol):
    url = f"https://finance.yahoo.com/quote/{ticker_symbol}/profile"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        leadership_data = []
        
        # Scrape leadership data
        profiles = soup.find_all("div", class_="Mb(25px)")
        for profile in profiles:
            name = profile.find("h3").text.strip() if profile.find("h3") else "N/A"
            title = profile.find("p", class_="Fz(s)").text.strip() if profile.find("p", class_="Fz(s)") else "N/A"
            leadership_data.append({
                "name": name,
                "title": title
            })
        return leadership_data
    else:
        print(f"Failed to fetch leadership data for {ticker_symbol}. Status code: {response.status_code}")
        return []

# Step 3: Fetch Latest News from Yahoo Finance
def fetch_latest_news(ticker_symbol):
    url = f"https://finance.yahoo.com/quote/{ticker_symbol}/news"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        news_headlines = []
        
        # Scrape news headlines
        articles = soup.find_all("h3", class_="Mb(5px)")
        for article in articles:
            headline = article.text.strip()
            news_headlines.append(headline)
        return news_headlines
    else:
        print(f"Failed to fetch news for {ticker_symbol}. Status code: {response.status_code}")
        return []

# Step 4: Generate Summary using DeepSeek
def generate_summary(financial_data, leadership_data, news_headlines):
    # Prepare the input prompt for DeepSeek
    prompt = f"""
    Analyze the following information about a company and provide a detailed summary:

    Financial Data:
    - P/E Ratio: {financial_data['pe_ratio']}
    - Revenue Growth: {financial_data['revenue_growth']}
    - Market Cap: {financial_data['market_cap']}

    Leadership Team:
    {', '.join([f"{member['name']} ({member['title']})" for member in leadership_data])}

    Latest News Headlines:
    {', '.join(news_headlines)}

    Provide a detailed summary of the company's fundamentals, leadership team, and recent news/actions. Highlight key insights and trends.
    """
    
    # Simulate calling DeepSeek (replace this with actual API calls if needed)
    summary = f"""
    The company has a P/E ratio of {financial_data['pe_ratio']}, which indicates its valuation relative to earnings. A revenue growth rate of {financial_data['revenue_growth']} suggests steady performance, while its market capitalization of {financial_data['market_cap']} reflects its significant presence in the market.

    The leadership team includes {', '.join([f"{member['name']} ({member['title']})" for member in leadership_data])}. This team brings a wealth of experience and expertise, which is crucial for the company's strategic direction and operational success.

    Recent news highlights include:
    - {', '.join(news_headlines)}.

    Overall, the company appears to be in a strong position, with solid financials, experienced leadership, and positive developments in the news. However, the high P/E ratio suggests that investors should exercise caution and monitor market conditions closely.
    """
    
    return summary

# Step 5: Main Function
def analyze_instrument(ticker_symbol):
    # Fetch financial data
    financial_data = fetch_financial_data(ticker_symbol)
    
    # Fetch leadership data
    leadership_data = fetch_leadership_data(ticker_symbol)
    
    # Fetch latest news
    news_headlines = fetch_latest_news(ticker_symbol)
    
    # Generate summary using DeepSeek
    summary = generate_summary(financial_data, leadership_data, news_headlines)
    
    # Display results
    print(f"\nAnalysis for {ticker_symbol}:")
    print(f"  P/E Ratio: {financial_data['pe_ratio']}")
    print(f"  Revenue Growth: {financial_data['revenue_growth']}")
    print(f"  Market Cap: {financial_data['market_cap']}")
    print("\nLeadership and Team:")
    for member in leadership_data:
        print(f"  {member['title']}: {member['name']}")
    print("\nLatest News Headlines:")
    for headline in news_headlines:
        print(f"  - {headline}")
    print("\nSummary:")
    print(summary)

# Main Program
if __name__ == "__main__":
    # Ask the user to enter a ticker symbol
    ticker = input("Enter the ticker symbol (e.g., AAPL, BTC-USD, TSLA): ").strip().upper()
    
    # Analyze the instrument
    analyze_instrument(ticker)