import matplotlib.pyplot as plt

# Test seti üzerinde tahmin yapalım ve sonuçları değerlendirelim
y_pred = model.predict(X_test)

# Grafik için boyutları ayarlayalım
plt.figure(figsize=(8, 6))

# Tahmin edilen ve gerçek değerleri bir grafikte gösterelim
plt.plot(y_test, color="red", label="Gerçek Değerler")
plt.plot(y_pred, color="blue", label="Tahmin Edilen Değerler")

# Grafik için etiketler ve başlık ekleyelim
plt.xlabel("Örnekler")
plt.ylabel("Hedef Değişken")
plt.title("Gerçek ve Tahmin Edilen Değerler")

# Grafik için gösterim ayarlarını yapılandıralım
plt.legend()
plt.show()
