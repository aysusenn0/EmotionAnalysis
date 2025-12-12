import torch
import gradio as gr
import os

# Ortak dosyadan modeli ve temizlik fonksiyonunu çekiyoruz
from model_utils import BiLSTM, clean_text


# MODELİ VE SÖZLÜĞÜ YÜKLEME

model_path = "final_model_data.pth"

if not os.path.exists(model_path):
    print("HATA: Model dosyası bulunamadı! Lütfen önce 'train.py' dosyasını çalıştırın.")
    exit()

print("Model yükleniyor...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Kaydettiğimiz paketi yüklüyoruz
checkpoint = torch.load(model_path, map_location=device)

# Sözlüğü ve parametreleri alıyoruz
vocab = checkpoint['vocab']
vocab_size = checkpoint['vocab_size']

model = BiLSTM(vocab_size)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()  # Değerlendirme modu

print("✅ Model ve Sözlük başarıyla yüklendi.")



def text_pipeline(text):
    # Metni temizle
    text = clean_text(text)
    tokens = text.split()
    # Yüklenen sözlüğü kullanarak sayıya çevir
    token_ids = [vocab.get(token, 1) for token in tokens]  # 1: <unk>

    MAX_LEN = 60
    if len(token_ids) < MAX_LEN:
        token_ids += [0] * (MAX_LEN - len(token_ids))  # 0: <pad>
    else:
        token_ids = token_ids[:MAX_LEN]

    return torch.tensor(token_ids, dtype=torch.long)


def tahmin_et(kullanici_yorumu):
    if not kullanici_yorumu: return "Lütfen bir yorum girin."

    # Veriyi hazırla
    vektor = text_pipeline(kullanici_yorumu).unsqueeze(0).to(device)

    # Tahmin
    with torch.no_grad():
        tahmin_skoru = model(vektor).item()
        olasilik = torch.sigmoid(torch.tensor(tahmin_skoru)).item()

    # Sonuç
    if olasilik > 0.50:  # Eşik değer
        sonuc = "OLUMLU 😊"
        renk = "green"
        guven = olasilik
    else:
        sonuc = "OLUMSUZ 😞"
        renk = "red"
        guven = 1 - olasilik

    return f"Tahmin: {sonuc}\nGüven Oranı: %{guven * 100:.2f}"



# GRADIO ARAYÜZÜ

print("Arayüz başlatılıyor...")

interface = gr.Interface(
    fn=tahmin_et,
    inputs=gr.Textbox(lines=2, placeholder="Yorumunuzu buraya yazın...", label="Müşteri Yorumu"),
    outputs=gr.Textbox(label="Yapay Zeka Analizi"),
    title="🛒 Ürün Yorumları Duygu Analizi",
    description="LSTM Modeli kullanılarak yorumun Olumlu mu yoksa Olumsuz mu olduğunu tahmin eder.",
    examples=[
        ["Ürün harika, çok beğendim, kargo hızlıydı."],
        ["Rezalet bir ürün, sakın almayın, hemen bozuldu."],
        ["Fiyatına göre idare eder."],
        ["Paketleme çok kötüydü ama ürün çalışıyor."]
    ]

)
if __name__ == "__main__":
    interface.launch(share=True)