import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# 입출력 폰트 설정
import platform
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # Mac OS
    plt.rc('font', family='AppleGothic')
else: # Linux
    # Linux에서는 별도의 폰트 설치 및 설정이 필요할 수 있습니다.
    # 예: plt.rc('font', family='NanumGothic')
    pass

# 마이너스 부호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# --------------------------------------------------------------------------
# 1. 데이터 수집 (수정된 부분)
# --------------------------------------------------------------------------
# 분석할 자산 티커와 이름 정의
tickers = {
    '^GSPC': 'S&P 500',
    'GLD': '금 (GLD ETF)',
    'BTC-USD': '비트코인',
    'TLT': '미국 장기채 (TLT ETF)'
}

# 분석 기간 설정
start_date = '2014-10-01'

asset_data_list = []

print("--------- [개별 티커 데이터 다운로드 시작] ---------")
for ticker_symbol, ticker_name in tickers.items():
    try:
        # 각 티커를 개별적으로 다운로드
        temp_df = yf.download(ticker_symbol, start=start_date, progress=False)
        
        # 다운로드 성공 여부 확인 (데이터가 비어있지 않다면)
        if not temp_df.empty:
            # 'Adj Close' 대신 'Close' 컬럼을 선택하도록 수정합니다.
            asset_data = temp_df[['Close']].rename(columns={'Close': ticker_symbol}) # <--- 수정된 부분
            asset_data_list.append(asset_data)
            print(f"✅ [{ticker_name}] 데이터 다운로드 성공")
        else:
            print(f"⚠️ [{ticker_name}] 데이터가 비어있어 제외합니다.")
            
    except Exception as e:
        print(f"❌ [{ticker_name}] 다운로드 중 에러 발생: {e}")

# 성공적으로 다운로드된 모든 데이터를 하나의 DataFrame으로 합치기
if asset_data_list:
    data = pd.concat(asset_data_list, axis=1)
    # 데이터가 모두 NaN인 행 제거 (주말 등)
    data.dropna(how='all', inplace=True)
    # 일부만 NaN인 행은 앞선 값으로 채우기 (선택적)
    data.fillna(method='ffill', inplace=True)

    print("\n--------- [최종 통합 데이터 확인] ---------")
    print(data.head())
    
    # ... 이 data 변수를 가지고 이후 분석 코드를 실행하면 됩니다 ...
    
else:
    print("\n모든 티커의 데이터를 다운로드하지 못했습니다.")

# --------------------------------------------------------------------------
# 2. 각 자산별 주가 선형회귀 분석 및 플로팅
# --------------------------------------------------------------------------
print("--------- [선형회귀 분석 시작] ---------")

for ticker_symbol, ticker_name in tickers.items():
    # 데이터 준비
    df_regr = data[[ticker_symbol]].copy()
    df_regr['time'] = np.arange(len(df_regr.index)) # 시간 변수(X) 생성

    X = df_regr[['time']]
    y = df_regr[ticker_symbol]

    # 선형회귀 모델 학습
    model = LinearRegression()
    model.fit(X, y)

    # 추세선 예측
    y_pred = model.predict(X)

    # 시각화
    plt.figure(figsize=(12, 6))
    plt.scatter(df_regr.index, y, s=1, label='실제 주가')
    plt.plot(df_regr.index, y_pred, color='red', linewidth=2, label='선형회귀 추세선')
    
    plt.title(f'[{ticker_name}] 주가와 선형회귀 추세선', fontsize=16)
    plt.xlabel('날짜', fontsize=12)
    plt.ylabel('주가 (USD)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

print("--------- [선형회귀 분석 완료] ---------\n")


# --------------------------------------------------------------------------
# 3. 누적 수익률 그래프 시각화
# --------------------------------------------------------------------------
print("--------- [누적 수익률 분석 시작] ---------")

# 일별 수익률 계산
returns = data.pct_change().dropna()

# 누적 수익률 계산 (초기 투자금을 1로 가정)
cumulative_returns = (1 + returns).cumprod()

# 시각화
plt.figure(figsize=(14, 7))
for ticker_symbol, ticker_name in tickers.items():
    plt.plot(cumulative_returns.index, cumulative_returns[ticker_symbol], label=ticker_name)

plt.title('자산별 누적 수익률 비교 (2018년 초=1)', fontsize=16)
plt.xlabel('날짜', fontsize=12)
plt.ylabel('누적 수익률', fontsize=12)
# 기준선(본전) 표시
plt.axhline(y=1, color='gray', linestyle='--', linewidth=1)
plt.legend()
plt.grid(True)
plt.show()

print("--------- [누적 수익률 분석 완료] ---------\n")


# --------------------------------------------------------------------------
# 4. 자산별 수익률 상관관계 매트릭스
# --------------------------------------------------------------------------
print("--------- [상관관계 분석 시작] ---------")

# 상관계수 행렬 계산
correlation_matrix = returns.corr()

# 결과 출력
print("[상관계수 행렬]")
# 컬럼명을 알아보기 쉽게 변경
corr_df_renamed = correlation_matrix.rename(columns=tickers, index=tickers)
print(corr_df_renamed)
print("\n")

# 히트맵으로 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(corr_df_renamed, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('자산별 수익률 상관관계 히트맵', fontsize=16)
plt.show()

print("--------- [상관관계 분석 완료] ---------")