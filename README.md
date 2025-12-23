# Türkçe Haber Kategorisi Sınıflandırma

Bu projede, **LSTM + CNN Hybrid** mimarisi kullanılarak Türkçe haber metinleri 4 farklı kategoriye sınıflandırılmaktadır:

- **Spor**
- **Ekonomi**
- **Dünya**
- **Güncel**

**Test Accuracy: 92.32%**

---

## Proje Konusu

### Motivasyon

Haber ajansları ve medya kuruluşları günlük olarak binlerce haber üretmektedir. Bu haberlerin manuel olarak kategorilere ayrılması zaman alıcı ve maliyetlidir. Otomatik metin sınıflandırma sistemleri:

- İçerik yönetimini kolaylaştırır
- Kullanıcılara kişiselleştirilmiş haber önerileri sunar
- Bilgiye erişimi hızlandırır
- Medya takibi ve analizi kolaylaştırır

### Literatür

Metin sınıflandırma alanında yapılan önemli çalışmalar:

- **Kim (2014)**: CNN ile metin sınıflandırma
- **Liu et al. (2016)**: Recurrent CNN (RCNN) modeli
- **Kılınç et al. (2017)**: TTC-3600 Türkçe benchmark veri seti
- **Toraman et al. (2022)**: BERTurk - Türkçe BERT modeli

Bu projede LSTM ve CNN'in hibrit kullanımıyla Türkçe haber sınıflandırmasında yüksek başarı (%92.32) elde edilmiştir.

---

## Veri Seti

### Kaynak

- **Veri Seti**: 42,000 Turkish News in 13 Classes
- **Kaynak**: YTÜ Kemik NLP Group
- **Platform**: [Kaggle](https://www.kaggle.com/datasets/oktayozturk010/42000-news-text-in-13-classes)

### Veri Seti Özellikleri

**Orijinal:**
- 42,000 haber metni
- 13 farklı kategori

**Kullanılan (Filtrelenmiş):**
- 22,833 haber metni
- 4 kategori (Spor, Güncel, Dünya, Ekonomi)
- "Genel" kategorisi çıkarıldı (belirsiz içerik)

### Kategori Dağılımı

| Kategori | Örnek Sayısı | Yüzde |
|----------|--------------|-------|
| Spor | 9,997 | 43.8% |
| Güncel | 5,847 | 25.6% |
| Dünya | 3,724 | 16.3% |
| Ekonomi | 3,265 | 14.3% |
| **Toplam** | **22,833** | **100%** |

### Veri Bölümü

| Bölüm | Örnek Sayısı | Yüzde |
|-------|--------------|-------|
| Eğitim | 16,496 | 72.2% |
| Validasyon | 2,912 | 12.8% |
| Test | 3,425 | 15.0% |

---

## Metodoloji

### Veri Ön İşleme

1. **Metin Temizleme:**
   - Küçük harfe çevirme
   - Sayı ve özel karakterlerin kaldırılması
   - Türkçe stopwords filtreleme (~250 kelime)
   - 2 karakterden kısa kelimelerin çıkarılması

2. **Tokenizasyon:**
   - Kelime dağarcığı: 15,000 kelime
   - Maksimum sequence uzunluğu: 250 token
   - Padding stratejisi: Post-padding

3. **Etiketleme:**
   - Label encoding ile numerik kodlama
   - Class weight balancing (dengesiz veri için)

### Model Eğitimi

**Hiperparametreler:**
- Optimizer: Adam (learning rate: 0.001)
- Loss Function: Sparse Categorical Crossentropy
- Batch Size: 32
- Epochs: 25 (Early stopping ile 10'da durdu)
- Embedding Dimension: 256

**Callbacks:**
- Early Stopping (patience=5)
- ReduceLROnPlateau (factor=0.5, patience=3)
- ModelCheckpoint (en iyi model kaydedildi)

**Düzenleme Teknikleri:**
- Spatial Dropout (0.3)
- Standard Dropout (0.3, 0.4)
- Batch Normalization
- Class Weight Balancing

---

### Mimari Seçim Gerekçesi

**LSTM Branch:**
- Sequential bilgiyi korur
- Uzun vadeli bağımlılıkları öğrenir
- Bidirectional yapı ile her iki yönden context

**CNN Branches:**
- N-gram pattern'lerini yakalar (3-gram ve 5-gram)
- Local features için etkili
- Hızlı özellik çıkarımı

**Hybrid Yaklaşım:**
- LSTM'in sequential gücü + CNN'in local pattern yakalama
- Çoklu kernel size ile farklı n-gram uzunlukları
- Robust ve yüksek performanslı sınıflandırma

---

## Sonuçlar

### Genel Performans

| Metrik | Değer |
|--------|-------|
| Test Accuracy | **92.32%** |
| Validation Accuracy | 92.72% |
| Macro Avg Precision | 0.9073 |
| Macro Avg Recall | 0.8985 |
| Macro Avg F1-Score | 0.9027 |
| Weighted Avg F1 | 0.9227 |

### Kategori Bazında Performans

| Kategori | Precision | Recall | F1-Score | Accuracy | Support |
|----------|-----------|--------|----------|----------|---------|
| Spor | 0.9719 | 0.9913 | 0.9815 | **99.13%** | 1499 |
| Güncel | 0.8783 | 0.8803 | 0.8793 | **88.03%** | 877 |
| Dünya | 0.8820 | 0.8694 | 0.8757 | **86.94%** | 559 |
| Ekonomi | 0.8970 | 0.8531 | 0.8745 | **85.31%** | 490 |

### Görselleştirmeler

#### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

### Karşılaştırmalı Analiz

| Model | Parametreler | Eğitim Süresi | Accuracy | F1-Score |
|-------|--------------|---------------|----------|----------|
| Naive Bayes | - | 1 dk | ~75% | 0.72 |
| SVM (TF-IDF) | - | 5 dk | ~82% | 0.80 |
| Simple LSTM | 500K | 2 dk | 72% | 0.69 |
| **LSTM + CNN (Bu Proje)** | **4.7M** | **5 dk** | **92.32%** | **0.92** |
| BERTurk | 110M | 30 dk | ~94% | 0.93 |

---

### Model Dosyalarını İndirme

Model dosyaları boyutları nedeniyle Google Drive'da barındırılmaktadır:

**[Model Dosyalarını İndir](https://drive.google.com/drive/folders/17e8JeBxM9qn14KZJxbhWzsPvc_lIe3zB?usp=sharing)**

İndirilen dosyaları `models/` klasörüne yerleştirin:
---

### Model Değerlendirmesi

```bash
# Tek bir metin için tahmin
python src/evaluate.py --text "Fenerbahçe Avrupa'da önemli bir galibiyet aldı"
```

**Çıktı:**
```
Predicted Category: spor
Confidence: 98.73%
```
