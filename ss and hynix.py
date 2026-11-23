import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# 한글 폰트 설정 (필요시)
import platform
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 다운로드
# 삼성전자, SK하이닉스 티커
tickers = {
    '005930.KS': '삼성전자',
    '000660.KS': 'SK하이닉스'
}
start_date = '2023-01-01'
end_date = '2024-12-31'

print("데이터 다운로드 중...")
data = yf.download(list(tickers.keys()), start=start_date, end=end_date)['Close']
data = data.rename(columns=tickers)
data.dropna(inplace=True)

# 2. 상관계수 계산 (일별 수익률 기준)
daily_returns = data.pct_change().dropna()
correlation = daily_returns['삼성전자'].corr(daily_returns['SK하이닉스'])

# 3. 주가 정규화 (시작일 = 100)
normalized_data = (data / data.iloc[0]) * 100

# 4. 시각화
fig, ax = plt.subplots(figsize=(12, 7))

# 정규화된 주가 선 차트 그리기
ax.plot(normalized_data.index, normalized_data['삼성전자'], label='삼성전자', linewidth=2, color='blue', alpha=0.7)
ax.plot(normalized_data.index, normalized_data['SK하이닉스'], label='SK하이닉스', linewidth=2, color='red', alpha=0.7)

# 차트 꾸미기
ax.set_title('삼성전자 - SK하이닉스 주가 추이 (2023~)', fontsize=16, fontweight='bold')
ax.set_ylabel('정규화된 주가 (시작일=100)', fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)

# 날짜 형식 설정
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# ★ 핵심: 상관계수 텍스트 추가 ★
# 그래프 오른쪽 하단 구석에 표시
text_str = f'Correlation: {correlation:.2f}'
# bbox는 텍스트 상자 테두리 및 배경 설정
ax.text(0.95, 0.05, text_str, transform=ax.transAxes, fontsize=15,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

plt.tight_layout()
plt.show()

print(f"\n[결과] 삼성전자와 SK하이닉스의 일별 수익률 상관계수: {correlation:.2f}")