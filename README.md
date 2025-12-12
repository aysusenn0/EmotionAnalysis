# 🇹🇷 Derin Öğrenme ile Türkçe Mağaza Yorumları Duygu Analizi Projesi

Bu proje, e-ticaret mağaza yorumlarını kullanarak metinlerin duygusal tonunu (Olumlu/Olumsuz) **Çift Yönlü Uzun-Kısa Süreli Bellek (BiLSTM)** sinir ağı ile sınıflandırmayı amaçlamaktadır. Proje, hem teknik uygulama hem de bilimsel raporlama kriterlerini karşılamak üzere geliştirilmiştir.

---

## 1. Proje Konusu ve Önemi 

### 1.1. Projenin Seçilme Gerekçesi ve İlgili Alanın Önemi
Güncel e-ticaret platformlarında, kullanıcı yorumlarının hacmi geleneksel analiz yöntemlerini aşmıştır. Proje, bu büyük veri yığınını otomatik olarak sınıflandırarak **Müşteri Geri Bildirimlerinin Anlık Analizi** için kritik bir araç sunar.

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
*  Öğrenme Oranı: 0.005
* **Epoch Sayısı:** 12
* **Eğitim Sonucu (Acc):** %95.04
*  **İlk learning rate 0.001'e göre çok daha iyi bir sonuç sergilediği için nihai seçimler seçildi.
* **Final Eğitim Kaybı (Loss)** | **0.135** |

### 4.2. Detaylı Model Değerlendirmesi (Test Verisi Üzerinden)

Modelin gerçek dünya performansını simüle eden test seti üzerindeki değerlendirmesi sonucunda, **%89.75** oranında genel doğruluk elde edilmiştir. Sınıf bazlı metrikler aşağıda detaylandırılmıştır.

#### [1] Detaylı Sınıflandırma Raporu (F1 Skoru)

| Metrik | Olumsuz (0) | Olumlu (1) | Weighted Avg (Ağırlıklı Ort.) |
| :--- | :--- | :--- | :--- |
| **Precision (Kesinlik)** | 0.8884 | 0.9064 | 0.8976 |
| **Recall (Duyarlılık)** | 0.9034 | 0.8918 | 0.8975 |
| **F1-Score** | **0.8958** | **0.8991** | **0.8975** |
| **Doğruluk (Accuracy)** | - | - | **0.8975** |

#### [2] Karışıklık Matrisi ve Hata Analizi

Modelin test setindeki 1697 yorum üzerindeki tahmin dağılımı şu şekildedir:

`Matris Çıktısı: [[748, 80], [94, 775]]`


> **Sonuç Analizi:** Model, Yanlış Negatif (94) ve Yanlış Pozitif (80) hataları arasında oldukça **dengeli bir dağılım** sergiledi.. İki hata türü arasındaki farkın az olması, modelin belirli bir sınıfa karşı (bias) önyargılı olmadığını ve genelleme yeteneğinin yüksek olduğunu kanıtlar.

## 5. Proje Dokümantasyonu ve Kod Düzeni 

Projenin yapısı, sürdürülebilirlik ve yeniden üretilebilirlik ilkelerine uygun olarak düzenlenmiştir:

* `train.py`: Modelin eğitimi, değerlendirilmesi ve modelin/sözlüğün kaydedilmesi.
* `model_utils.py`: Model mimarisi (`BiLSTM`) ve veri temizleme (`clean_text`) fonksiyonlarının merkezi.
* `model_serve.py`: Eğitilmiş modelin Gradio ile web arayüzünde sunulması.
* `final_model_data.pth`: Eğitilmiş model ağırlıklarını ve kelime sözlüğünü içerir.
* `README.md`: Bu dokümantasyon, projenin tüm aşamalarını ve sonuçlarını açıklar.

---
# EN Sentiment Analysis of Turkish Store Reviews with Deep Learning

This project performs sentiment classification (Positive/Negative) on Turkish e-commerce product reviews using a **Bidirectional Long Short-Term Memory (BiLSTM)** deep learning model.

---

## 1. Project Topic and Importance

### 1.1. Motivation  
The rapid growth of customer reviews on online platforms exceeds the limits of manual analysis. Automated sentiment classification supports **real-time customer feedback monitoring**, which is crucial for decision-making in e-commerce systems.

### 1.2. Comparison with Existing Approaches  
- **Traditional ML (SVM, Naive Bayes):** Fast but fails to capture word order and context.  
- **Deep Learning:** Learns sequential dependencies and achieves higher accuracy.

---

## 2. Dataset and Preprocessing

### 2.1. Dataset  
- **Source:** Kaggle – “Duygu Analizi İçin Ürün Yorumları”  
- **Link:** https://www.kaggle.com/datasets/burhanbilenn/duygu-analizi-icin-urun-yorumlari/data  
- **Size:** 8,484 Turkish product reviews  

### 2.2. Preprocessing  
- Binary label assignment (Positive / Negative)  
- Removal of numbers, punctuation, and special characters  
- Turkish stop-word removal  
- Text vectorization into integer sequences  
- **Vocabulary size:** 4,002  

---

## 3. Methodology

### 3.1. Selected Method: Bidirectional LSTM (BiLSTM)  
BiLSTM processes sequences in both forward and backward directions, allowing better understanding of contextual transitions.

Example:  
*“Ürün hızlıydı fakat kalitesi hayal kırıklığıydı.”*  
The contrast introduced by *“fakat”* is captured more effectively by BiLSTM.

---

## 4. Model Training and Evaluation

### 4.1. Training Summary  
- **Model:** BiLSTM  
- **Hidden Size:** 128  
- **Optimizer:** Adam  
- **Learning Rate:** 0.005  
- **Epochs:** 12  
- **Training Accuracy:** 95.04%  
- **Final Loss:** 0.135  

### 4.2. Test Results  

#### Classification Report

| Metric | Negative (0) | Positive (1) | Weighted Avg |
|--------|--------------|--------------|--------------|
| Precision | 0.8884 | 0.9064 | 0.8976 |
| Recall | 0.9034 | 0.8918 | 0.8975 |
| F1-Score | 0.8958 | 0.8991 | 0.8975 |
| Accuracy | – | – | **0.8975** |

#### Confusion Matrix  : 
[[748, 80], [94, 775]]

**Interpretation:**  
False Positives (80) and False Negatives (94) are balanced, indicating that the model is not biased toward any class.

