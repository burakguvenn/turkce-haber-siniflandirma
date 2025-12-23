"""
Gradio Demo Arayüzü
Türkçe Haber Sınıflandırma Modeli
"""

import gradio as gr
import numpy as np
import pickle
import re
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk

# NLTK stopwords
nltk.download('stopwords', quiet=True)

# Model ve preprocessing araçlarını yükle
print("Model yükleniyor...")
model = keras.models.load_model('../models/news_classifier_hybrid.keras')

with open('../models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('../models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print("Model başarıyla yüklendi.")

# Türkçe stopwords
turkish_stopwords = set(stopwords.words('turkish'))
extra_stopwords = {
    'dedi', 'söyledi', 'açıkladı', 'belirtti', 'kaydetti', 'etti', 'olan',
    'üzere', 'göre', 'karşı', 'önce', 'sonra', 'bugün', 'dün', 'yarın'
}
turkish_stopwords.update(extra_stopwords)

MAX_LEN = 250


def preprocess_text(text):
    """Metin ön işleme"""
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-züğışöçıİ\s]', ' ', text)
    text = ' '.join(text.split())
    
    words = text.split()
    words = [w for w in words if w not in turkish_stopwords and len(w) > 2]
    
    return ' '.join(words)


def predict_category(text):
    """Kategori tahmini yap"""
    if not text or text.strip() == "":
        return "Lütfen bir haber metni girin.", {}
    
    # Metni temizle
    clean_text = preprocess_text(text)
    
    if len(clean_text) < 10:
        return "Metin çok kısa, lütfen daha uzun bir haber metni girin.", {}
    
    # Sequence'e çevir
    sequence = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Tahmin yap
    prediction = model.predict(padded, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    
    # Sonuçları hazırla
    category = label_encoder.classes_[predicted_class]
    
    # Kategori isimleri
    category_names = {
        'spor': 'Spor',
        'ekonomi': 'Ekonomi',
        'dunya': 'Dünya',
        'guncel': 'Güncel'
    }
    
    # Tüm kategoriler için güven skorları
    results = {}
    for i, cat in enumerate(label_encoder.classes_):
        cat_name = category_names.get(cat, cat)
        results[cat_name] = float(prediction[0][i])
    
    # Güven seviyesi
    if confidence > 0.9:
        confidence_level = "Çok Yüksek"
    elif confidence > 0.75:
        confidence_level = "Yüksek"
    elif confidence > 0.6:
        confidence_level = "Orta"
    else:
        confidence_level = "Düşük"
    
    # Ana sonuç
    result_text = f"""
## Tahmin Edilen Kategori: {category_names.get(category, category).upper()}

Güven Skoru: {confidence*100:.2f}% ({confidence_level} Güven)
"""
    
    return result_text, results


# Örnek metinler
examples = [
    ["Galatasaray, UEFA Avrupa Ligi'nde Manchester United'ı 3-1 mağlup ederek tarihi bir galibiyet aldı. Icardi'nin iki golü maçın yıldızı oldu."],
    ["Fenerbahçe Beko, THY Euroleague'de Real Madrid'i deplasmanda yenerek Play-Off şansını artırdı. Vesely ve Motley harika performans sergiledi."],
    ["Türkiye Cumhuriyet Merkez Bankası politika faizini yüzde 45'e yükseltti. Enflasyonla mücadelede kararlılık mesajı verildi."],
    ["Borsa İstanbul BIST 100 endeksi tüm zamanların en yüksek seviyesi olan 10 bin 500 puanı aştı. Yabancı yatırımcı ilgisi arttı."],
    ["ABD Başkanı Biden, Çin Devlet Başkanı Xi Jinping ile kritik bir görüşme gerçekleştirdi. İki ülke arasındaki ticaret anlaşmazlıkları masaya yatırıldı."],
    ["Japonya'da 7.2 büyüklüğünde deprem meydana geldi. Tsunami uyarısı yapıldı, binlerce kişi tahliye edildi. Hasar bilgileri toplanıyor."],
    ["İstanbul'da yoğun sis nedeniyle tüm deniz ulaşımı iptal edildi. Vatandaşlar alternatif ulaşım araçlarına yöneldi, trafik kilitlendi."],
    ["Sağlık Bakanlığı, grip salgınına karşı vatandaşları aşı olmaya çağırdı. Hastanelerde yoğunluk yaşanıyor, önlem alınması isteniyor."]
]

# Gradio arayüzü
with gr.Blocks(title="Türkçe Haber Kategorisi Sınıflandırma") as demo:
    gr.Markdown("""
    # Türkçe Haber Kategorisi Sınıflandırma
    ### LSTM + CNN Hybrid Model
    
    **Kategoriler:** Spor | Ekonomi | Dünya | Güncel
    
    Bir haber metni girin ve hangi kategoriye ait olduğunu öğrenin.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Haber Metni",
                placeholder="Buraya haber metnini yazın veya alttaki örneklerden birini seçin...",
                lines=10
            )
            with gr.Row():
                clear_btn = gr.Button("Temizle", size="sm")
                predict_btn = gr.Button("Tahmin Et", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            result_output = gr.Markdown(label="Tahmin Sonucu")
            confidence_output = gr.Label(label="Tüm Kategoriler için Güven Skorları", num_top_classes=4)
    
    gr.Markdown("### Örnek Haberler")
    gr.Examples(
        examples=examples,
        inputs=text_input,
        label="Aşağıdaki örnek haber metinlerinden birini seçebilirsiniz"
    )
    
    gr.Markdown("""
    ---
    **Model Bilgisi:**
    - Algoritma: Bidirectional LSTM + CNN Hybrid
    - Test Accuracy: 92.32%
    - Eğitim verisi: ~23,000 Türkçe haber
    - 4 kategori: Spor, Ekonomi, Dünya, Güncel
    """)
    
    predict_btn.click(
        fn=predict_category,
        inputs=text_input,
        outputs=[result_output, confidence_output]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", {}),
        inputs=None,
        outputs=[text_input, result_output, confidence_output]
    )


if __name__ == "__main__":
    demo.launch(share=True)