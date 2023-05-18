
import pandas as pd
df=pd.read_csv(r"C:\Users\kader\OneDrive\Masaüstü\CV\wheat3.csv",parse_dates=['Tarih'], index_col='Tarih')
df=df.dropna()
print('Shape of data',df.shape)
df.head()
df