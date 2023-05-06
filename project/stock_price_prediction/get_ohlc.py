import yfinance as yf

df = yf.download('^N225', period='max', interval='1d')

df.loc[df.index > '2015-01-01', 'Close'].plot()
df.to_csv('N225.csv')
