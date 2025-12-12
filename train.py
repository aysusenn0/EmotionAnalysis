import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import time

# Ortak dosyadan modeli ve temizlik fonksiyonunu çekiyoruz
from model_utils import BiLSTM, clean_text

path = r"C:\Users\AYSU\.cache\kagglehub\datasets\burhanbilenn\duygu-analizi-icin-urun-yorumlari\versions\1"
dosya_adi = "magaza_yorumlari_duygu_analizi.csv"
full_path = os.path.join(path, dosya_adi)

print(f"--- Veri Yükleniyor ---")

try:
    df = pd.read_csv(full_path, encoding='utf-16', sep=',', quotechar='"', on_bad_lines='skip')
except:
    try:
        df = pd.read_csv(full_path, encoding='utf-16', sep='\t', quotechar='"', on_bad_lines='skip')
    except:
        files = [f for f in os.listdir(path) if f.endswith('.csv')]
        full_path = os.path.join(path, files[0])
        df = pd.read_csv(full_path, encoding='utf-16', sep=',', quotechar='"', on_bad_lines='skip')


df = df.iloc[:, [0, -1]]
df.columns = ['Yorum', 'Durum']

# Temizlik
df = df[df['Durum'].isin(['Olumlu', 'Olumsuz'])].copy()
df['Etiket'] = df['Durum'].apply(lambda x: 1.0 if x == 'Olumlu' else 0.0)
df['Yorum_Temiz'] = df['Yorum'].apply(clean_text)
df = df[df['Yorum_Temiz'].str.len() > 2]

print(f"Eğitim için Veri Sayısı: {len(df)}")


# HAZIRLIK (Sözlük Oluşturma)

X = df['Yorum_Temiz'].values
y = df['Etiket'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%20'sini test için ayırdım


print("Sözlük oluşturuluyor...")

all_text = " ".join(X_train)
words = all_text.split()
word_counts = Counter(words)
common_words = word_counts.most_common(4000)

vocab = {word: i + 2 for i, (word, count) in enumerate(common_words)}
vocab['<pad>'] = 0
vocab['<unk>'] = 1
VOCAB_SIZE = len(vocab)
print(f"Sözlük Boyutu: {VOCAB_SIZE}")


# Dataset Hazırlığı
def text_pipeline_train(text, vocabulary):
    tokens = text.split()
    token_ids = [vocabulary.get(token, 1) for token in tokens]
    MAX_LEN = 60
    if len(token_ids) < MAX_LEN:
        token_ids += [0] * (MAX_LEN - len(token_ids))
    else:
        token_ids = token_ids[:MAX_LEN]
    return torch.tensor(token_ids, dtype=torch.long)


class SentimentDataset(Dataset):
    def __init__(self, X, y, vocab):
        self.X = X
        self.y = y
        self.vocab = vocab

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        return text_pipeline_train(self.X[idx], self.vocab), torch.tensor(self.y[idx], dtype=torch.float32)


BATCH_SIZE = 32
train_loader = DataLoader(SentimentDataset(X_train, y_train, vocab), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(SentimentDataset(X_test, y_test, vocab), batch_size=BATCH_SIZE, shuffle=False)

#EĞİTİM

model = BiLSTM(VOCAB_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005) #0.005 learning rate
criterion = nn.BCEWithLogitsLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)

EPOCHS = 12 #12 deneme
print(f"\n--- Eğitim Başlıyor ({device}) ---")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for text, label in train_loader:
        text, label = text.to(device), label.to(device)
        optimizer.zero_grad()
        predictions = model(text).squeeze(1)
        loss = criterion(predictions, label)

        rounded_preds = torch.round(torch.sigmoid(predictions))
        correct = (rounded_preds == label).float()
        acc = correct.sum() / len(correct)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    print(
        f"Epoch {epoch + 1:02} | Loss: {epoch_loss / len(train_loader):.3f} | Acc: {epoch_acc / len(train_loader) * 100:.2f}%")

#MODEL DEĞERLENDİRİLMESİ


print("\n--- Modelin Test Verisi Üzerindeki Detaylı Değerlendirilmesi ---")

# Modeli değerlendirme moduna al
model.eval()

all_preds = []
all_labels = []

# Gradyan hesaplamasını devre dışı bırak
with torch.no_grad():
    for text, label in test_loader:
        text, label = text.to(device), label.to(device)

        # Tahminleri al
        predictions = model(text).squeeze(1)

        # Sigmoid uygulayıp 0.5 eşiği ile yuvarla (logit -> 0 veya 1)
        rounded_preds = torch.round(torch.sigmoid(predictions))

        # Sonuçları CPU'ya taşıyıp listelere ekle
        all_preds.extend(rounded_preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())


all_preds = np.array(all_preds).astype(int)
all_labels = np.array(all_labels).astype(int)

# 1. SINIFLANDIRMA RAPORU (Precision, Recall, F1-Score)

print("\n[1] Detaylı Sınıflandırma Raporu:")
print(classification_report(all_labels, all_preds, target_names=['Olumsuz (0)', 'Olumlu (1)'], digits=4))

# 2. KARIŞIKLIK MATRİSİ
# Modelin ne tür hatalar yaptığını (Yanlış Pozitif/Negatif) gösterir.
print("\n[2] Karışıklık Matrisi (Hata Türleri Analizi):")
cm = confusion_matrix(all_labels, all_preds)
print(cm)

"""
Karışıklık Matrisi Yapısı :
[[TN  FP]
 [FN  TP]]
... (Açıklamalar) ...
"""

# Model_serve.py dosyasının çalışabilmesi için hem modeli hem de kelime sözlüğünü (vocab) kaydetmeliyiz.
print("\nModel ve Sözlük kaydediliyor...")

save_data = {
    'model_state_dict': model.state_dict(),
    'vocab': vocab,
    'vocab_size': VOCAB_SIZE
}

torch.save(save_data, "final_model_data.pth")
print("✅ BAŞARILI: 'final_model_data.pth' dosyası oluşturuldu.")