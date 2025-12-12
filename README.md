# 🇹🇷 Derin Öğrenme ile Türkçe Mağaza Yorumları Duygu Analizi Projesi

Bu proje, e-ticaret mağaza yorumlarını kullanarak metinlerin duygusal tonunu (Olumlu/Olumsuz) **Çift Yönlü Uzun-Kısa Süreli Bellek (BiLSTM)** sinir ağı ile sınıflandırmayı amaçlamaktadır. Proje, hem teknik uygulama hem de bilimsel raporlama kriterlerini karşılamak üzere geliştirilmiştir.

---

## 1. Proje Konusu ve Önemi 

### 1.1. Projenin Seçilme Gerekçesi ve İlgili Alanın Önemi
Güncel e-ticaret platformlarında, kullanıcı yorumlarının hacmi geleneksel analiz yöntemlerini aşmıştır. Proje, bu büyük veri yığınını otomatik olarak sınıflandırarak **Müşteri Geri Bildirimlerinin Anlık Analizi** için kritik bir araç sunar. Bu, işletmelerin ürün kalitesini hızla değerlendirmesi ve marka itibarını koruması açısından hayati önem taşır.

### 1.2. İlgili Alanda Yapılan Uygulamalara Karşılaştırmalı Bakış
Duygu analizi, Makine Öğrenmesi (ML) ve Derin Öğrenme (DL) tekniklerinin kesişim noktasındadır.
* **Geleneksel ML (SVM, Naive Bayes):** Hızlıdır ancak kelimelerin sırasını ve dolayısıyla **bağlamı** kaybeder.
* **Derin Öğrenme (DL):** Metin dizilerindeki karmaşık ve uzun vadeli bağımlılıkları öğrenerek daha yüksek doğruluk sağlar.

---

## 2. Veri Setinin Belirlenmesi ve Ön İşleme 

### 2.1. Veri Seti
* **Kaynak:** Kaggle, "Duygu Analizi İçin Ürün Yorumları"
* https://www.kaggle.com/datasets/burhanbilenn/duygu-analizi-icin-urun-yorumlari/data
* **Boyut:** Toplam 8484 adet yorum (eğitim ve test için ayrılmıştır).

### 2.2. Ön İşleme ve Temizlik
1.  **Sınıflandırma:** Yorumlar, ikili sınıflandırma (Olumlu/Olumsuz) için etiketlenmiştir.
2.  **Metin Temizliği:** Sayılar, noktalama işaretleri ve özel karakterler kaldırılmıştır.
3.  **Kritik Adım:** Modelin anlamlı kelimelere odaklanması için **Türkçe Stop Word (durma kelimesi) kaldırma** işlemi uygulanmıştır.
4.  **Vektörleştirme:** Yorumlar, modelin anlayabileceği sayısal dizilere dönüştürülmüş ve sözlük boyutu 4002 olarak belirlenmiştir.

---

## 3. Uygulanacak Yöntem/Algoritmanın Seçim Gerekçesi

### Seçilen Yöntem: Çift Yönlü LSTM (BiLSTM)

BiLSTM, metin dizilerinde **bağlamı yakalama** konusunda Tek Yönlü LSTM ve geleneksel ML'e göre üstünlük sağlar.

* **Tek Yönlü LSTM vs. BiLSTM:** Tek Yönlü LSTM, bir kelimeyi yalnızca kendinden **önceki** kelimelere bakarak yorumlarken, BiLSTM hem **ileri** hem de **geri** yönde (cümle sonundan başına) bilgi akışı sağlar. 
* **Avantajı:** "Ürün hızlıydı **fakat** kalitesi hayal kırıklığıydı." gibi cümlelerdeki "fakat" gibi zıtlık bildiren bağlaçların öncesi ve sonrası arasındaki kritik bilgiyi, BiLSTM etkin bir şekilde öğrenir. Bu, duygu analizi için en dengeli ve yüksek performanslı çözümü sunar.

---

## 4. Model Eğitimi & Model Değerlendirilmesi 

### 4.1. Model Eğitimi Özeti
* **Model Mimarisi:** BiLSTM (Gizli Katman Boyutu: 128)
* **Optimizasyon:** Adam Optimizer
*  Öğrenme Oranı: 0.001
* **Epoch Sayısı:** 12
* **Eğitim Sonucu (Acc):** %90.65

### 4.2. Detaylı Model Değerlendirmesi (Test Verisi Üzerinden)

Modelin nihai performansı, akademik çalışmalarda standart olan **F1 Skoru** ve **Karışıklık Matrisi** ile değerlendirilmiştir.

#### [1] Detaylı Sınıflandırma Raporu (F1 Skoru)
| Metrik | Olumsuz (0) | Olumlu (1) | Weighted Avg (Ağırlıklı Ortalama) |
| :--- | :--- | :--- | :--- |
| **Precision** | 0.8664 | 0.9226 | 0.8952 |
| **Recall** | 0.9239 | 0.8642 | 0.8933 |
| **F1-Score** | **0.8942** | **0.8925** | **0.8933** |
| **Doğruluk (Accuracy)** | | | **0.8933** |

#### [2] Karışıklık Matrisi ve Hata Analizi
Matris çıktısı: `[[765 63], [118 751]]`

| Değer | Tanım | Analiz |
| :--- | :--- | :--- |
| **FN (118)** | Yanlış Negatif (Gerçekte Olumlu, Tahmin: Olumsuz) | Modelin en sık yaptığı hata: Olumlu yorumları kaçırma. |
| **FP (63)** | Yanlış Pozitif (Gerçekte Olumsuz, Tahmin: Olumlu) | Modelin yanlışlıkla iyimser olduğu durumlar (daha az). |

---

## 5. Proje Dokümantasyonu ve Kod Düzeni 

Projenin yapısı, sürdürülebilirlik ve yeniden üretilebilirlik ilkelerine uygun olarak düzenlenmiştir:

* `train.py`: Modelin eğitimi, değerlendirilmesi ve modelin/sözlüğün kaydedilmesi.
* `model_utils.py`: Model mimarisi (`BiLSTM`) ve veri temizleme (`clean_text`) fonksiyonlarının merkezi.
* `model_serve.py`: Eğitilmiş modelin Gradio ile web arayüzünde sunulması.
* `final_model_data.pth`: Eğitilmiş model ağırlıklarını ve kelime sözlüğünü içerir.
* `README.md`: Bu dokümantasyon, projenin tüm aşamalarını ve sonuçlarını açıklar.

---

