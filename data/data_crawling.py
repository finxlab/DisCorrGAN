import yfinance as yf
import pandas as pd

# 데이터 가져올 종목의 티커 심볼 설정
tickers = {
    'DJI': '^DJI',
    'Nasdaq': '^IXIC',
    'JP Morgan': 'JPM',
    'Hang Seng': '^HSI',
    'Gold': 'GC=F',
    'WTI': 'CL=F'
}

# 시작일과 종료일 설정
start_date = '2001-01-01'
end_date = '2024-12-31'

# 데이터를 저장할 빈 데이터프레임 생성
data = pd.DataFrame()

for name, ticker in tickers.items():
    print(f"{name} ({ticker}) 데이터 다운로드 중...")
    
    # 'Adj Close'를 포함하도록 auto_adjust=False로 설정
    stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

        
    column_name = 'Adj Close'
    stock_data = stock_data[[column_name]].rename(columns={column_name: name})

    # 데이터프레임 병합
    if data.empty:
        data = stock_data
    else:
        data = data.join(stock_data, how='outer')

# 결측치 처리
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)
data = data[(data >= 0).all(axis=1)]

# 데이터 확인
print(data.head())

# CSV 파일로 저장
data.to_csv('indices.csv')
