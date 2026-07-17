#  Deteksi Penyakit Daun pada Tanaman Padi

## Deskripsi Proyek

Proyek ini merupakan implementasi **Kecerdasan Buatan** untuk mendeteksi penyakit pada daun tanaman padi menggunakan algoritma **Decision Tree** dan **K-Nearest Neighbors (KNN)**. Sistem ini dirancang untuk membantu petani dan penyuluh pertanian dalam mengidentifikasi jenis penyakit daun padi secara cepat dan akurat berdasarkan citra daun.

**Topik:** Deteksi Penyakit Daun pada Tanaman Padi Menggunakan Algoritma Decision Tree dan K-Nearest Neighbors (KNN)

**Mata Kuliah:** Kecerdasan Buatan
---

## 📁 Struktur Folder
UAS-KecerdasanBuatan/
├── README.md # Deskripsi proyek
├── Laporan UAS.md # Laporan lengkap
├── uas_model.ipynb # Notebook utama
├── data/
│ └── [dataset] # Folder dataset
├── Hasil/
│ ├── Distribusi jumlah Gambar per kelas.png
│ ├── Gambar Daun Padi Per Kelas.png
│ ├── Distribusi Kelas Data train & Test.png
│ └── Confusion Matrix Desicion Tree & KNN.png
└── Jurnal/
├── jurnal1.pdf
├── jurnal2.pdf
├── jurnal3.pdf
├── jurnal4.pdf
└── jurnal5.pdf

---

## 🎯 Tujuan Proyek

1. Mengembangkan model klasifikasi penyakit daun padi menggunakan algoritma **Decision Tree** dan **K-Nearest Neighbors (KNN)**
2. Membandingkan kinerja kedua algoritma dalam mengklasifikasikan jenis penyakit daun padi
3. Menentukan algoritma terbaik berdasarkan metrik evaluasi (akurasi, presisi, recall, F1-score)

---

## 🗂️ Dataset

### Sumber Data
Dataset yang digunakan berasal dari **Kaggle - Rice Plant Diseases Dataset** oleh jay7080dev (2023).

### Detail Dataset
| Aspek | Keterangan |
|-------|------------|
| **Total Gambar** | 4.740 gambar |
| **Jumlah Kelas** | 3 kelas |
| **Format** | JPG/PNG |

### Kelas Penyakit
| Kelas | Jumlah Gambar | Persentase |
|-------|---------------|------------|
| **Bacterialblight** | 1.580 | 33.3% |
| **Brownspot** | 1.580 | 33.3% |
| **Leafsmut** | 1.580 | 33.3% |

### Cara Mendapatkan Dataset
1. Buka [Kaggle - Rice Plant Diseases Dataset](https://www.kaggle.com/datasets/jay7080dev/rice-plant-diseases-dataset)
2. Klik tombol **Download**
3. Ekstrak file zip ke folder `data/`

---

## 🔧 Teknologi yang Digunakan

| Teknologi | Keterangan |
|-----------|------------|
| **Python 3.10** | Bahasa pemrograman |
| **Google Colab** | Platform eksekusi |
| **Scikit-learn** | Library machine learning |
| **OpenCV** | Pengolahan citra |
| **Scikit-image** | Ekstraksi fitur tekstur (LBP) |
| **Matplotlib & Seaborn** | Visualisasi data |
| **Pandas & NumPy** | Manipulasi data |

---

## 📊 Fitur yang Diekstraksi

| Jenis Fitur | Metode | Jumlah Fitur |
|-------------|--------|--------------|
| **Fitur Warna** | RGB Histogram | 48 |
| **Fitur Tekstur** | Local Binary Pattern (LBP) | 10 |
| **Fitur Bentuk** | Geometric Features | 3 |
| **Total** | | **61** |

---

## 🤖 Algoritma yang Digunakan

### 1. Decision Tree
- Parameter: `criterion='gini'`, `max_depth=10`, `random_state=42`
- Kelebihan: Mudah diinterpretasi, menangani data non-linear
- Kekurangan: Rentan overfitting

### 2. K-Nearest Neighbors (KNN)
- Parameter: `n_neighbors=5`, `metric='euclidean'`, `weights='distance'`
- Kelebihan: Sederhana, tidak perlu training
- Kekurangan: Lambat untuk dataset besar, sensitif skala

---

## 📈 Hasil Evaluasi

### Perbandingan Metrik

| Metrik | Decision Tree | KNN |
|--------|---------------|-----|
| **Accuracy** | **90.48%** | 88.17% |
| **Precision** | **90.51%** | 88.20% |
| **Recall** | **90.48%** | 88.17% |
| **F1-Score** | **90.48%** | 88.15% |

### 🏆 Model Terbaik
**Decision Tree** dengan akurasi **90.48%**

### Alasan Decision Tree Menjadi Model Terbaik
1. Akurasi tertinggi (90.48%)
2. Konsisten di semua kelas penyakit
3. Mudah diinterpretasikan
4. Tidak terlalu sensitif terhadap skala data

### Confusion Matrix Decision Tree

| | Prediksi BLB | Prediksi BS | Prediksi LS |
|---|--------------|-------------|-------------|
| **Aktual BLB** | 318 | 21 | 3 |
| **Aktual BS** | 21 | 322 | 4 |
| **Aktual LS** | 24 | 28 | 292 |

---

## 🚀 Cara Menjalankan

### Opsi 1: Google Colab (Direkomendasikan)
1. Buka [Google Colab](https://colab.research.google.com/)
2. Upload file `uas_model.ipynb`
3. Jalankan semua cell secara berurutan

### Opsi 2: Lokal (VS Code / Jupyter)
1. Clone repository:
   ```bash
   git clone https://github.com/username/UAS-KecerdasanBuatan.git
   cd UAS-KecerdasanBuatan

    Hasil Visualisasi
Gambar	Keterangan
Distribusi jumlah Gambar per kelas.png	Distribusi jumlah gambar per kelas
Gambar Daun Padi Per Kelas.png	Contoh gambar dari setiap kelas
Distribusi Kelas Data train & Test.png	Distribusi data training dan testing
Confusion Matrix Desicion Tree & KNN.png	Confusion matrix kedua model

