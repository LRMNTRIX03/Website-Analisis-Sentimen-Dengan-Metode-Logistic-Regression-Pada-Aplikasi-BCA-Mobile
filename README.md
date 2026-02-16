# Sentiment Analysis Web App (Django)

Website ini adalah sistem analisis sentimen berbasis teks yang dibangun dengan **Django**, yang memproses data ulasan aplikasi **BCA Mobile** untuk mengklasifikasi sentimen secara otomatis. Proyek ini mendukung pengolahan data secara massal melalui file **CSV**, serta pengujian sentimen berdasarkan input kalimat manual.

---

## Fitur Utama

- **Upload Data Review via CSV**
- **Preprocessing Otomatis** (case folding, simbol, slangword, stopword, stemming)
- **Upload Stopword & Slangword via CSV**
- **Labelisasi Otomatis** menggunakan **TextBlob**
- Translate otomatis ke Bahasa Inggris menggunakan **deep_translator** / Google Translator
- **Klasifikasi Sentimen** menggunakan Logistic Regression / model ML lainnya
- **Testing Kalimat** manual (contoh: review dari pengguna BCA Mobile)

---

## Teknologi & Library

- **Backend**: Django
- **Natural Language Toolkit**: NLTK, Sastrawi
- **ML**: scikit-learn, TextBlob
- **Translator**: deep_translator (GoogleTranslator)
- **Frontend**: Tailwindcss, Alpine.js, dan DaisyUI

## Tata Cara Menggunakan

Ikuti langkah berikut untuk menjalankan project ini di lokal:

---
1. Clone Project
```bash
git clone https://github.com/username/nama-repo.git](https://github.com/LRMNTRIX03/Website-Analisis-Sentimen-Dengan-Metode-Logistic-Regression-Pada-Aplikasi-BCA-Mobile
cd Website-Analisis-Sentimen-Dengan-Metode-Logistic-Regression-Pada-Aplikasi-BCA-Mobile
```
2. Buat Environment
## Windows
```bash
python -m venv venv
venv\Scripts\activate
```
## Mac/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Setelah aktif instanll library python
```bash
pip install -r requirements.txt
```
4. Konfigurasi Database di Settings.py sesuaikan dengan db_anda
5. Migrasi Database
```bash
python manage.py migrate
```
6. Buat superuser Admin
```bash
python manage.py createsuperuser
```
7. Jalankan server
```bash
python manage.py runserver
```
8. Buka server anda
http://127.0.0.1:8000/
