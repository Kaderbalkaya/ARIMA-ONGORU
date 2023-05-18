# Gerekli kütüphaneleri yükleyelim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Veri kümesini yükleyelim
#data = pd.read_csv(r"C:\Users\kader\OneDrive\Masaüstü\CV\wheat1-1.csv", names=["Year", "Production"], sep=';')

file_path = r"C:\Users\kader\OneDrive\Masaüstü\CV\wheat1-1.csv"
df = pd.read_csv(file_path)

# Yıl ve üretim adedini seç
X = df.iloc[:, 0:1].values
y = df.iloc[:, 1].values

# Doğrusal regresyon modelini eğit
model = LinearRegression()
model.fit(X, y)

# Yeni bir yıl için üretim adedi tahmin et
yeni_yil = [[2033]]
yeni_uretim = model.predict(yeni_yil)
print("2033 yılında öngörülen üretim adedi: ", yeni_uretim)

# Gerçek verileri ve tahminleri gösteren bir grafik oluştur
y_pred = model.predict(X)
plt.scatter(X, y, color='red')
plt.plot(X, y_pred, color='blue')
plt.title('Yıllara Göre Üretim Miktarı')
plt.xlabel('Yıl')
plt.ylabel('Üretim Miktarı')
plt.show()