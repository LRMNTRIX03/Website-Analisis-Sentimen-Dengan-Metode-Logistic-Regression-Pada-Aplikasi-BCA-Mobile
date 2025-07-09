# 🧠 Sentiment Analysis Web App (Django)

Website ini adalah sistem analisis sentimen berbasis teks yang dibangun dengan **Django**, yang memproses data ulasan aplikasi **BCA Mobile** untuk mengklasifikasi sentimen secara otomatis. Proyek ini mendukung pengolahan data secara massal melalui file **CSV**, serta pengujian sentimen berdasarkan input kalimat manual.

---

## 🚀 Fitur Utama

- 🔄 **Upload Data Review via CSV**
- 🧹 **Preprocessing Otomatis** (case folding, simbol, slangword, stopword, stemming)
- 🧾 **Upload Stopword & Slangword via CSV**
- 🌐 **Labelisasi Otomatis** menggunakan **TextBlob**
  - Translate otomatis ke Bahasa Inggris menggunakan **deep_translator** / Google Translator
- 🤖 **Klasifikasi Sentimen** menggunakan Logistic Regression / model ML lainnya
- 🔍 **Testing Kalimat** manual (contoh: review dari pengguna BCA Mobile)

---

## 🛠️ Teknologi & Library

- **Backend**: Django
- **Natural Language Toolkit**: NLTK, Sastrawi
- **ML**: scikit-learn, TextBlob
- **Translator**: deep_translator (GoogleTranslator)
- **Frontend**: Bootstrap (jika pakai)

---

## 📂 Struktur Fitur Utama

