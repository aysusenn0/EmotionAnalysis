import torch
import torch.nn as nn
import re
import pandas as pd


# 1. Ortak Metin Temizleme Fonksiyonu
def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    # Sadece harfleri ve boşlukları bırak
    text = re.sub(r'[^a-zıüöçşğ\s]', '', text)
    return text.strip()


# 2. Model Mimarisi (Her iki tarafın da bunu bilmesi lazım)
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=64, output_dim=1):
        super(BiLSTM, self).__init__()

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True) #çift yönlü seçim

        # Fully Connected (Çift yönlü olduğu için hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(0.5)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)

        # İleri ve Geri yönlü son durumları birleştir
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        return self.fc(self.dropout(hidden_cat))