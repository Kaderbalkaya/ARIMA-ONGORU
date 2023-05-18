import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.specification import SARIMAXSpecification
from sklearn.metrics import mean_squared_error,mean_absolute_error
#barley için eklendi
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)



#Veri kümesini yükleyelim
data=pd.read_csv(r"C:\Users\kader\OneDrive\Masaüstü\CV\rye.csv")


#Veri kümesi tarih ve üretim miktarı sütunları 
data['Tarih']=pd.to_datetime(data['Tarih'], format='%Y')
data.set_index('Tarih',inplace=True)
y=data['Uretim Miktari']


#ARIMA model oluşturma ve eğitme 
model= sm.tsa.statespace.SARIMAX(y, order=(0, 1, 0), seasonal_order=(0,1,1,6), enforce_stationarity=False )
model_fit=model.fit()

#Öngörü tahmini yapalım
y_pred=model_fit.forecast(steps=10)
print(y_pred)


#Sonuçları grafik hale getirelim
plt.figure(figsize=(10,5))
plt.plot(y,label="Gerçek değerler")
plt.plot(y_pred,label="Tahmin edilen değer")
plt.xlabel("Yillar")
plt.ylabel("Üretim Miktari")
plt.title("Üretim Miktari Ongoru")
plt.legend()
plt.show()

# Yıllara göre öngörü tahmini yapalım
y_by_year = data.resample('Y').sum()['Uretim Miktari']
model = sm.tsa.statespace.SARIMAX(y_by_year, order=(0, 1, 0))
model_fit = model.fit()
y_pred_by_year = model_fit.forecast(steps=10)
print(y_pred_by_year)

#Başarı oranı bulma MSE (Mean Squared Error) veya RMSE (Root Mean Squared Error) hesaplama
mse = mean_squared_error(y_pred, y_pred_by_year)
mae = mean_absolute_error(y_pred, y_pred_by_year)
rmse = np.sqrt(mse)

print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')

# Yıllara göre öngörü tahminini grafikle gösterelim
plt.figure(figsize=(10, 5))
plt.plot(y_by_year, label="Gerçek Değerler")
plt.plot(y_pred_by_year, label="Tahmin Edilen Değerler")
plt.xlabel("Yıl")
plt.ylabel("Üretim Miktarı")
plt.title("Yıllara Göre Üretim Miktarı Öngörüsü")
plt.legend()
plt.show()