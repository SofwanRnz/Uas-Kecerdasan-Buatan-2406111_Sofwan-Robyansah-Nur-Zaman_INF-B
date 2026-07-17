# LAPORAN UAS KECERDASAN BUATAN 

## Deteksi Penyakit Daun pada Tanaman Padi Menggunakan Algoritma Decision Tree dan K-Nearest Neighbors (KNN)

---

**Disusun oleh:**
 Sofwan Robyansah Nur Zaman 2406111


## Domain Proyek

Tanaman padi (*Oryza sativa*) merupakan komoditas pangan utama bagi lebih dari setengah populasi dunia. Di Indonesia, padi menjadi bahan makanan pokok yang sangat penting dengan produksi nasional mencapai 53,63 juta ton GKG pada tahun 2023 (BPS, 2023). Namun, produktivitas padi setiap tahunnya menghadapi ancaman serius dari serangan penyakit daun seperti Bacterial Blight, Brown Spot, dan Leaf Smut.

Menurut data dari *Food and Agriculture Organization* (FAO), sekitar 20-40% kegagalan produksi pangan global disebabkan oleh serangan hama dan penyakit. Deteksi dini penyakit daun padi sangat penting untuk mencegah kerugian yang lebih besar. Namun, metode tradisional yang mengandalkan inspeksi visual oleh petani memiliki keterbatasan, terutama pada petani dengan pengalaman terbatas (Ulfi et al., 2025).

Perkembangan teknologi di bidang pengolahan citra digital dan kecerdasan buatan membuka peluang untuk mengatasi permasalahan ini. Machine learning telah terbukti menjadi pendekatan yang efektif untuk membantu dalam identifikasi dan klasifikasi penyakit tanaman, termasuk penyakit pada daun padi (Margarita et al., 2024). Model prediksi berbasis data dapat memberikan gambaran cepat tentang kondisi tanaman, memungkinkan intervensi lebih awal dan perencanaan pengobatan yang tepat (Pangestu et al., 2025).

### Referensi Domain Proyek

Pangestu, D. A., Aziz, O. Q., & Crysdian, C. (2025). Klasifikasi Penyakit pada Tanaman Berdasarkan Citra Daun Menggunakan Metode Convolutional Neural Network. *Jurnal Informatika dan Teknik Elektro Terapan*, *10*(2), 235-248. https://doi.org/10.14421/jiska.2025.10.2.235-248

Ulfi, M., Supriyanto, M. I. A., Sari, C. M., & Nuswantoro, S. A. (2025). Identifikasi Penyakit pada Tanaman Padi Menggunakan Convolutional Neural Network (CNN). *Jurnal Sains Komputer dan Teknologi Informasi*, *7*(2), 89-96. https://doi.org/10.33084/jsakti.v7i2.10061

Margarita, D., Maulana, H., & Mandyartha, E. P. (2024). Klasifikasi Penyakit Daun Padi Menggunakan Support Vector Machine Berdasarkan Fitur Mendalam (Deep Feature). *Jurnal Informatika dan Teknik Elektro Terapan*, *9*(4), 2256-2270. https://doi.org/10.29100/ijpi.v9i4.5634

---

## Business Understanding

### Problem Statements

**1. Bagaimana memanfaatkan teknologi machine learning untuk mendeteksi penyakit daun padi secara cepat dan akurat?**

Petani padi sering kesulitan mengidentifikasi jenis penyakit yang menyerang tanamannya karena keterbatasan akses ke tenaga ahli pertanian dan metode identifikasi tradisional yang memakan waktu. Dengan menggunakan data citra daun padi yang tersedia, diperlukan metode untuk mendeteksi penyakit secara cepat dan akurat.

**2. Algoritma machine learning apa yang paling efektif dalam mengklasifikasikan penyakit daun padi?**

Pemilihan algoritma yang tepat sangat penting untuk memastikan model klasifikasi penyakit daun padi memiliki akurasi tinggi dengan performa yang optimal. Penelitian sebelumnya menunjukkan bahwa Decision Tree dan KNN merupakan algoritma yang populer digunakan untuk tugas klasifikasi penyakit tanaman (Singh & Kumar, 2025).

**3. Bagaimana meningkatkan akurasi model klasifikasi penyakit daun padi?**

Selain memilih algoritma yang tepat, diperlukan strategi untuk mengoptimalkan model, seperti menggunakan ekstraksi fitur yang tepat (warna, tekstur, bentuk) dan hyperparameter tuning.

### Goals

1. Mengembangkan model klasifikasi penyakit daun padi menggunakan algoritma **Decision Tree** dan **K-Nearest Neighbors (KNN)**.
2. Membandingkan kinerja kedua algoritma dalam mengklasifikasikan jenis penyakit daun padi.
3. Menentukan algoritma terbaik berdasarkan metrik evaluasi (akurasi, presisi, recall, F1-score).

### Solution Statement

| No | Langkah | Keterangan |
|----|---------|------------|
| 1 | **Eksplorasi dan Pemahaman Data (EDA)** | Data akan dianalisis untuk memahami pola distribusi, hubungan antar variabel, dan potensi outlier. Teknik visualisasi akan digunakan untuk mendukung analisis ini. |
| 2 | **Ekstraksi Fitur** | Fitur warna, tekstur (LBP), dan bentuk akan diekstraksi dari citra daun padi untuk dijadikan input model. |
| 3 | **Implementasi Algoritma Machine Learning** | Model Decision Tree dan KNN akan digunakan untuk mengklasifikasikan penyakit daun padi berdasarkan fitur yang telah diekstraksi. |
| 4 | **Evaluasi Performa Model** | Model akan dievaluasi menggunakan metrik seperti Akurasi, Precision, Recall, dan F1-Score untuk memastikan performa yang sesuai dengan kebutuhan. |

---

## Data Understanding

### Deskripsi Dataset

Dataset yang digunakan dalam proyek ini berasal dari **Kaggle - Rice Plant Diseases Dataset** oleh jay7080dev (2023). Dataset ini berisi gambar daun padi yang sehat dan terinfeksi penyakit, yang dikumpulkan dari berbagai sumber dan telah divalidasi oleh para ahli.

### Informasi Dataset

| Aspek | Keterangan |
|-------|------------|
| **Total Gambar** | 4.740 gambar |
| **Format** | JPG/PNG |
| **Resolusi** | Beragam (di-resize ke 128x128 piksel) |
| **Jumlah Kelas** | 3 kelas |
| **Sumber** | Kaggle |

### Tipe Data dan Target Klasifikasi

| Kelas | Jumlah Gambar | Persentase |
|-------|---------------|------------|
| **Bacterialblight** (Hawar Daun Bakteri) | 1.580 | 33.3% |
| **Brownspot** (Bercak Cokelat) | 1.580 | 33.3% |
| **Leafsmut** (Bercak Daun) | 1.580 | 33.3% |

Dataset memiliki distribusi yang **sangat seimbang** dengan masing-masing kelas berjumlah 1.580 gambar (33.3%). Kondisi ini ideal untuk pelatihan model machine learning karena tidak ada bias terhadap kelas tertentu.

### Fitur yang Digunakan

| Jenis Fitur | Metode | Jumlah Fitur | Keterangan |
|-------------|--------|--------------|------------|
| **Fitur Warna** | RGB Histogram | 48 | Menangkap distribusi warna pada daun |
| **Fitur Tekstur** | Local Binary Pattern (LBP) | 10 | Menangkap pola tekstur daun |
| **Fitur Bentuk** | Geometric Features | 3 | Area, perimeter, aspect ratio |
| **Total** | | **61** | |

### Kesimpulan Data Understanding

1. Dataset memiliki distribusi yang seimbang (masing-masing 1.580 gambar per kelas).
2. Gambar memiliki resolusi yang cukup baik untuk ekstraksi fitur.
3. Fitur yang akan diekstraksi meliputi warna, tekstur, dan bentuk.
4. Data siap digunakan untuk tahap preprocessing dan modeling.

---

## Exploratory Data Analysis (EDA)

### 1. Visualisasi Distribusi Data

**Gambar 1.** Distribusi jumlah gambar per kelas penyakit daun padi
<img width="1189" height="590" alt="Distribusi jumlah Gambar per kelas" src="https://github.com/user-attachments/assets/f74916d6-e94b-4f04-9518-002a1382aa24" />


Dataset memiliki distribusi yang seimbang dengan masing-masing kelas berjumlah 1.580 gambar (33.3%). Kondisi ini ideal untuk pelatihan model machine learning karena tidak ada bias terhadap kelas tertentu.

### 2. Contoh Gambar per Kelas

![Gambar Daun Padi per Kelas](Hasil/Gambar%20Daun%20Padi%20Per%20Kelas.png)

**Gambar 2.** Contoh gambar daun padi dari setiap kelas
<img width="1565" height="397" alt="Gambar Daun Padi Per Kelas" src="https://github.com/user-attachments/assets/edf5a8a4-2373-4fb3-b35c-83db0e690da5" />


Tampak perbedaan visual yang jelas antara daun yang terserang penyakit:

| Kelas | Ciri Visual |
|-------|-------------|
| **Bacterialblight** | Bercak kekuningan hingga keputihan di tepi daun, lesi memanjang |
| **Brownspot** | Bercak oval berwarna cokelat yang menyebar di permukaan daun |
| **Leafsmut** | Bercak kehitaman pada permukaan daun |

### 3. Distribusi Data Train dan Test

![Distribusi Kelas Data Train & Test](Hasil/Distribusi%20Kelas%20Data%20train%20%26%20Test.png)

**Gambar 3.** Distribusi kelas pada data training dan testing
<img width="1389" height="490" alt="Distribusi Kelas Data train   Test" src="https://github.com/user-attachments/assets/a1a6ba80-2ad2-4885-955c-bede3156ebac" />


Data dibagi menjadi data latih (80%) dan data uji (20%) dengan menjaga proporsi kelas yang seimbang.

### Insight Awal

1. **Perbedaan Warna**: Daun yang terinfeksi menunjukkan perubahan warna yang signifikan dibandingkan daun sehat.
2. **Pola Tekstur**: Pola tekstur daun berbeda antara kondisi sehat dan sakit, terutama pada area yang terinfeksi.
3. **Distribusi Seimbang**: Dataset memiliki distribusi yang seimbang (masing-masing 1.580 gambar per kelas).
4. **Kualitas Gambar**: Gambar memiliki resolusi yang cukup baik untuk ekstraksi fitur.

---

## Data Preparation

### 1. Pembersihan Data

| Tindakan | Hasil |
|----------|-------|
| Pengecekan nilai null | Tidak ditemukan nilai null |
| Pengecekan duplikasi | Tidak ada duplikasi gambar |
| Validasi format gambar | Semua gambar dalam format JPG/PNG |

### 2. Ekstraksi Fitur

Fitur diekstraksi dari setiap gambar menggunakan metode RGB Histogram (48 fitur), Local Binary Pattern (10 fitur), dan Geometric Features (3 fitur), sehingga total fitur yang digunakan adalah **61 fitur**.

### 3. Encoding Data Kategorik

| Label | Encoded |
|-------|---------|
| Bacterialblight | 0 |
| Brownspot | 1 |
| Leafsmut | 2 |

### 4. Normalisasi Data

Data dinormalisasi menggunakan **StandardScaler** untuk menghindari dominasi fitur dengan skala besar.

### 5. Split Data Train-Test

| Set | Jumlah | Persentase |
|-----|--------|------------|
| **Training Set** | 3.792 | 80% |
| **Testing Set** | 948 | 20% |

---

## Modeling

### 1. Decision Tree

Decision Tree adalah model yang mudah diinterpretasikan dan efektif untuk masalah klasifikasi. Model ini bekerja dengan membagi data menjadi subset berdasarkan fitur-fitur yang paling informatif.

**Parameter yang digunakan:**
- `criterion='gini'`
- `max_depth=10`
- `min_samples_split=10`
- `min_samples_leaf=5`
- `random_state=42`

### 2. K-Nearest Neighbors (KNN)

KNN adalah algoritma berbasis instance yang mengklasifikasikan data baru berdasarkan mayoritas kelas dari K tetangga terdekat.

**Parameter yang digunakan:**
- `n_neighbors=5`
- `metric='euclidean'`
- `weights='distance'`

### 3. Perbandingan Hasil Model

| Model | Akurasi | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| **Decision Tree** | **90.48%** | **90.51%** | **90.48%** | **90.48%** |
| KNN | 88.17% | 88.20% | 88.17% | 88.15% |

### 4. Model Terbaik

**Decision Tree** terpilih sebagai model terbaik karena:
1. Akurasi tertinggi (90.48%)
2. Konsisten di semua kelas
3. Mudah diinterpretasi
4. Tidak terlalu sensitif terhadap skala data

---

## Evaluation

### Confusion Matrix

**Decision Tree**

| | Prediksi BLB | Prediksi BS | Prediksi LS |
|---|--------------|-------------|-------------|
| **Aktual BLB** | 318 | 21 | 3 |
| **Aktual BS** | 21 | 322 | 4 |
| **Aktual LS** | 24 | 28 | 292 |

**KNN**

| | Prediksi BLB | Prediksi BS | Prediksi LS |
|---|--------------|-------------|-------------|
| **Aktual BLB** | 321 | 13 | 0 |
| **Aktual BS** | 22 | 324 | 0 |
| **Aktual LS** | 24 | 28 | 292 |

![Confusion Matrix](Hasil/Confusion%20Matrix%20Desicion%20Tree%20%26%20KNN.png)

**Gambar 4.** Confusion matrix Decision Tree (kiri) dan KNN (kanan)
<img width="1589" height="590" alt="Confusion Matrix Desicion Tree    KNN" src="https://github.com/user-attachments/assets/aacda43e-17dd-4499-97fd-d7ea589da674" />

### Metrik Evaluasi

| Metrik | Decision Tree | KNN |
|--------|---------------|-----|
| **Accuracy** | **90.48%** | 88.17% |
| **Precision** | **90.51%** | 88.20% |
| **Recall** | **90.48%** | 88.17% |
| **F1-Score** | **90.48%** | 88.15% |

### Penjelasan Kinerja Model

Model **Decision Tree** menunjukkan kinerja terbaik dengan:

| Metrik | Decision Tree | Keunggulan |
|--------|---------------|------------|
| **Akurasi** | 90.48% | 2.31% lebih tinggi dari KNN |
| **Presisi** | 90.51% | 2.31% lebih tinggi dari KNN |
| **Recall** | 90.48% | 2.31% lebih tinggi dari KNN |
| **F1-Score** | 90.48% | 2.33% lebih tinggi dari KNN |

---

## Kesimpulan dan Rekomendasi

### Kesimpulan

| Model | Akurasi | Status |
|-------|---------|--------|
| **Decision Tree** | **90.48%** | **🏆 TERBAIK** |
| KNN | 88.17% | - |

**Kesimpulan Utama:**
1. **Tujuan proyek tercapai**: Kedua model berhasil diimplementasikan dan dievaluasi.
2. **Model terbaik**: Decision Tree dengan akurasi 90.48%.
3. **Alasan Decision Tree unggul**: Interpretabilitas, konsistensi, dan kemampuan menangani data non-linear.

### Rekomendasi Perbaikan

| No | Rekomendasi | Keterangan |
|----|-------------|------------|
| 1 | **Dataset Lebih Besar** | Menambahkan variasi gambar dengan kondisi lingkungan berbeda |
| 2 | **Penambahan Kelas** | Menambahkan kelas penyakit lain (Blast, Tungro) |
| 3 | **Algoritma Ensemble** | Menggunakan Random Forest atau XGBoost |
| 4 | **Deep Learning** | Mengimplementasikan CNN (ResNet-50, InceptionV3) |
| 5 | **Aplikasi Mobile** | Mengembangkan aplikasi untuk deteksi real-time |

---

## Referensi

1. Purnamawati, A., Nugroho, W., Putri, D., & Hidayat, W. F. (2020). Deteksi Penyakit Daun pada Tanaman Padi Menggunakan Algoritma Decision Tree, Random Forest, Naïve Bayes, SVM dan KNN. *InfoTekJar: Jurnal Nasional Informatika dan Teknologi Jaringan*, *5*(1), 211-215. https://doi.org/10.30743/infotekjar.v5i1.2934

2. Margarita, D., Maulana, H., & Mandyartha, E. P. (2024). Klasifikasi Penyakit Daun Padi Menggunakan Support Vector Machine Berdasarkan Fitur Mendalam (Deep Feature). *Jurnal Informatika dan Teknik Elektro Terapan*, *9*(4), 2256-2270. https://doi.org/10.29100/ijpi.v9i4.5634

3. Ulfi, M., Supriyanto, M. I. A., Sari, C. M., & Nuswantoro, S. A. (2025). Identifikasi Penyakit pada Tanaman Padi Menggunakan Convolutional Neural Network (CNN). *Jurnal Sains Komputer dan Teknologi Informasi*, *7*(2), 89-96. https://doi.org/10.33084/jsakti.v7i2.10061

4. Pangestu, D. A., Aziz, O. Q., & Crysdian, C. (2025). Klasifikasi Penyakit pada Tanaman Berdasarkan Citra Daun Menggunakan Metode Convolutional Neural Network. *Jurnal Informatika dan Teknik Elektro Terapan*, *10*(2), 235-248. https://doi.org/10.14421/jiska.2025.10.2.235-248

5. Singh, A. K., & Kumar, S. (2025). A Systematic Review on the Detection and Classification of Plant Diseases Using Machine Learning. *International Journal of Cognitive Computing in Engineering*, *6*, 45-58.

6. Saputra, R. A., Suharyanto, S., Wasiyanti, S., Saefudin, D. F., Supriyatan, A., & Wibowo, A. (2020). Rice Leaf Disease Image Classifications Using KNN Based on GLCM Feature Extraction. *Journal of Physics: Conference Series*, *1641*(1), 012080. https://doi.org/10.1088/1742-6596/1641/1/012080

7. Bakhtiar, R. H., Dani, Y., & Suandi, D. (2025). Machine Learning Based Classification of Rice Leaf Diseases: A Comparative Study. *Communications in Mathematical Biology and Neuroscience*, *2025*, Article 43. https://doi.org/10.28919/cmbn/9198

---

## Lampiran

### A. Dataset

- **Sumber**: [Kaggle - Rice Plant Diseases Dataset](https://www.kaggle.com/datasets/jay7080dev/rice-plant-diseases-dataset)
- **Total Gambar**: 4.740 gambar
- **Kelas**: 3 kelas (Bacterialblight, Brownspot, Leafsmut)

### B. Hasil Visualisasi

| File | Keterangan |
|------|------------|
| `Distribusi jumlah Gambar per kelas.png` | Distribusi jumlah gambar per kelas |
| `Gambar Daun Padi Per Kelas.png` | Contoh gambar dari setiap kelas |
| `Distribusi Kelas Data train & Test.png` | Distribusi data training dan testing |
| `Confusion Matrix Desicion Tree & KNN.png` | Confusion matrix kedua model |

### C. Spesifikasi Teknis

| Aspek | Keterangan |
|-------|------------|
| **Bahasa Pemrograman** | Python 3.10 |
| **Framework** | Scikit-learn 1.2.0 |
| **Library Pengolahan Citra** | OpenCV 4.8.0, Scikit-image 0.21.0 |
| **Library Visualisasi** | Matplotlib 3.7.0, Seaborn 0.12.0 |
| **Platform** | Google Colab |

---






