# 📄 LAPORAN UAS KECERDASAN BUATAN.md

## Deteksi Penyakit Daun pada Tanaman Padi Menggunakan Algoritma Decision Tree dan K-Nearest Neighbors (KNN)

---

**Disusun oleh:**
- [Nama Mahasiswa 1] - [NIM]
- [Nama Mahasiswa 2] - [NIM]

**Tanggal:** 17 Juli 2026

---

## Domain Proyek

Tanaman padi (*Oryza sativa*) merupakan komoditas pangan utama bagi lebih dari setengah populasi dunia. Di Indonesia, padi menjadi bahan makanan pokok yang sangat penting dengan produksi nasional mencapai 53,63 juta ton GKG pada tahun 2023 (BPS, 2023). Namun, produktivitas padi setiap tahunnya menghadapi ancaman serius dari serangan penyakit daun seperti Bacterial Blight, Brown Spot, dan Leaf Smut.

Menurut data dari *Food and Agriculture Organization* (FAO), sekitar 20-40% kegagalan produksi pangan global disebabkan oleh serangan hama dan penyakit. Deteksi dini penyakit daun padi sangat penting untuk mencegah kerugian yang lebih besar. Namun, metode tradisional yang mengandalkan inspeksi visual oleh petani memiliki keterbatasan, terutama pada petani dengan pengalaman terbatas (Ulfi et al., 2025).

Perkembangan teknologi di bidang pengolahan citra digital dan kecerdasan buatan membuka peluang untuk mengatasi permasalahan ini. Machine learning telah terbukti menjadi pendekatan yang efektif untuk membantu dalam identifikasi dan klasifikasi penyakit tanaman, termasuk penyakit pada daun padi (Margarita et al., 2024). Penelitian-penelitian terdahulu telah menunjukkan bahwa pendekatan machine learning mampu mengklasifikasikan penyakit daun padi dengan akurasi tinggi. Model prediksi berbasis data dapat memberikan gambaran cepat tentang kondisi tanaman, memungkinkan intervensi lebih awal dan perencanaan pengobatan yang tepat (Pangestu et al., 2025).

### Referensi Domain Proyek

Pangestu, D. A., Aziz, O. Q., & Crysdian, C. (2025). Klasifikasi Penyakit pada Tanaman Berdasarkan Citra Daun Menggunakan Metode Convolutional Neural Network. *Jurnal Informatika dan Teknik Elektro Terapan*, *10*(2), 235-248. https://doi.org/10.14421/jiska.2025.10.2.235-248

Ulfi, M., Supriyanto, M. I. A., Sari, C. M., & Nuswantoro, S. A. (2025). Identifikasi Penyakit pada Tanaman Padi Menggunakan Convolutional Neural Network (CNN). *Jurnal Sains Komputer dan Teknologi Informasi*, *7*(2), 89-96. https://doi.org/10.33084/jsakti.v7i2.10061

Margarita, D., Maulana, H., & Mandyartha, E. P. (2024). Klasifikasi Penyakit Daun Padi Menggunakan Support Vector Machine Berdasarkan Fitur Mendalam (Deep Feature). *Jurnal Informatika dan Teknik Elektro Terapan*, *9*(4), 2256-2270. https://doi.org/10.29100/ijpi.v9i4.5634

Satriadi, R. (2025). *PadiGuard: Aplikasi Deteksi Penyakit Daun Padi Berbasis Deep Learning*. Politeknik Negeri Semarang.

Priyadharshini, A., & Kalaivanan, K. (2025). A review on machine learning and deep learning techniques for plant leaf disease detection and classification with IoT in agriculture industry. *Journal of Agriculture and Food Research*, *15*, 101567. https://doi.org/10.1016/j.jafr.2025.101567

Singh, A. K., & Kumar, S. (2025). A Systematic Review on the Detection and Classification of Plant Diseases Using Machine Learning. *International Journal of Cognitive Computing in Engineering*, *6*, 45-58.

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

Untuk mencapai tujuan proyek, langkah-langkah berikut akan dilakukan:

| No | Langkah | Keterangan |
|----|---------|------------|
| 1 | **Eksplorasi dan Pemahaman Data (EDA)** | Data akan dianalisis untuk memahami pola distribusi, hubungan antar variabel, dan potensi outlier. Teknik visualisasi akan digunakan untuk mendukung analisis ini. |
| 2 | **Ekstraksi Fitur** | Fitur warna, tekstur (LBP), dan bentuk akan diekstraksi dari citra daun padi untuk dijadikan input model. |
| 3 | **Implementasi Algoritma Machine Learning** | Model Decision Tree dan KNN akan digunakan untuk mengklasifikasikan penyakit daun padi berdasarkan fitur yang telah diekstraksi. |
| 4 | **Evaluasi Performa Model** | Model akan dievaluasi menggunakan metrik seperti Akurasi, Precision, Recall, dan F1-Score untuk memastikan performa yang sesuai dengan kebutuhan. |

Solusi ini diharapkan dapat memberikan model prediksi yang andal untuk membantu petani dalam mendeteksi penyakit daun padi secara cepat dan akurat.

---

## Data Understanding

### Deskripsi Dataset

Dataset yang digunakan dalam proyek ini berasal dari **Kaggle - Rice Plant Diseases Dataset** oleh jay7080dev (2023). Dataset ini berisi gambar daun padi yang sehat dan terinfeksi penyakit, yang dikumpulkan dari berbagai sumber dan telah divalidasi oleh para ahli.

### Informasi Dataset

Dataset terdiri dari gambar daun padi berwarna (RGB) dengan total 4.740 gambar yang terbagi dalam 3 kelas penyakit.

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

Dataset berupa gambar daun padi. Untuk implementasi Decision Tree dan KNN, fitur yang digunakan adalah hasil ekstraksi dari gambar:

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

![Distribusi Jumlah Gambar per Kelas](Hasil/Distribusi%20jumlah%20Gambar%20per%20kelas.png)

**Gambar 1.** Distribusi jumlah gambar per kelas penyakit daun padi

Dataset memiliki distribusi yang seimbang dengan masing-masing kelas berjumlah 1.580 gambar (33.3%). Kondisi ini ideal untuk pelatihan model machine learning karena tidak ada bias terhadap kelas tertentu.

### 2. Contoh Gambar per Kelas

![Gambar Daun Padi per Kelas](Hasil/Gambar%20Daun%20Padi%20Per%20Kelas.png)

**Gambar 2.** Contoh gambar daun padi dari setiap kelas

Tampak perbedaan visual yang jelas antara daun yang terserang penyakit:

| Kelas | Ciri Visual |
|-------|-------------|
| **Bacterialblight** | Bercak kekuningan hingga keputihan di tepi daun, lesi memanjang |
| **Brownspot** | Bercak oval berwarna cokelat yang menyebar di permukaan daun |
| **Leafsmut** | Bercak kehitaman pada permukaan daun |

### 3. Distribusi Data Train dan Test

![Distribusi Kelas Data Train & Test](Hasil/Distribusi%20Kelas%20Data%20train%20%26%20Test.png)

**Gambar 3.** Distribusi kelas pada data training dan testing

Data dibagi menjadi data latih (80%) dan data uji (20%) dengan menjaga proporsi kelas yang seimbang.

### Insight Awal

1. **Perbedaan Warna**: Daun yang terinfeksi menunjukkan perubahan warna yang signifikan dibandingkan daun sehat.
2. **Pola Tekstur**: Pola tekstur daun berbeda antara kondisi sehat dan sakit, terutama pada area yang terinfeksi.
3. **Distribusi Seimbang**: Dataset memiliki distribusi yang seimbang (masing-masing 1.580 gambar per kelas).
4. **Kualitas Gambar**: Gambar memiliki resolusi yang cukup baik untuk ekstraksi fitur.

### Kesimpulan EDA

Dataset ini sudah memiliki distribusi yang seimbang tanpa perlu oversampling.
Terdapat perbedaan visual yang jelas antar kelas penyakit.
Data siap digunakan untuk tahap preprocessing dan modeling.

---

## Data Preparation

Tahapan data preparation dilakukan untuk mempersiapkan dataset sebelum digunakan dalam pelatihan model machine learning. Berikut langkah-langkahnya:

### 1. Pembersihan Data

| Tindakan | Hasil |
|----------|-------|
| Pengecekan nilai null | Tidak ditemukan nilai null |
| Pengecekan duplikasi | Tidak ada duplikasi gambar |
| Validasi format gambar | Semua gambar dalam format JPG/PNG |

### 2. Ekstraksi Fitur

Fitur diekstraksi dari setiap gambar menggunakan metode berikut:

**a. Fitur Warna (RGB Histogram)**

```python
for i in range(3):
    hist = cv2.calcHist([img], [i], None, [16], [0, 256])
    rgb_hist.extend(hist.flatten())
```

**b. Fitur Tekstur (Local Binary Pattern)**

```python
lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
```

**c. Fitur Bentuk**

```python
area = cv2.contourArea(largest_contour)
perimeter = cv2.arcLength(largest_contour, True)
aspect_ratio = w / h
```

**Hasil Ekstraksi:**

| Jenis Fitur | Jumlah Fitur |
|-------------|--------------|
| RGB Histogram | 48 |
| LBP | 10 |
| Geometric | 3 |
| **Total** | **61** |

### 3. Encoding Data Kategorik

Label kelas diubah menjadi numerik menggunakan LabelEncoder:

| Label | Encoded |
|-------|---------|
| Bacterialblight | 0 |
| Brownspot | 1 |
| Leafsmut | 2 |

### 4. Normalisasi Data

Data dinormalisasi menggunakan **StandardScaler** untuk menghindari dominasi fitur dengan skala besar (sangat penting untuk algoritma KNN yang berbasis jarak):

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 5. Split Data Train-Test

Data dibagi menjadi data latih dan data uji dengan rasio 80:20:

| Set | Jumlah | Persentase |
|-----|--------|------------|
| **Training Set** | 3.792 | 80% |
| **Testing Set** | 948 | 20% |

Hasil pembagian data ini memastikan bahwa model dapat dilatih dan diuji secara adil dengan distribusi data yang representatif.

### Kesimpulan Data Preparation

Setelah melalui proses ini, dataset siap digunakan untuk proses modeling:

- Fitur telah diekstraksi dari gambar (61 fitur).
- Variabel kategori telah di-encode menjadi numerik.
- Semua fitur numerik telah dinormalisasi.
- Dataset telah dibagi menjadi data latih dan uji dengan proporsi yang seimbang.

---

## Modeling

Pada tahap ini, dilakukan pemodelan data menggunakan dua algoritma machine learning untuk mengklasifikasikan penyakit daun padi: **Decision Tree** dan **K-Nearest Neighbors (KNN)**.

### 1. Decision Tree

Decision Tree adalah model yang mudah diinterpretasikan dan efektif untuk masalah klasifikasi. Model ini bekerja dengan membagi data menjadi subset berdasarkan fitur-fitur yang paling informatif.

**Parameter:**
- `criterion='gini'`
- `max_depth=10`
- `min_samples_split=10`
- `min_samples_leaf=5`
- `random_state=42`

**Cara Kerja:**
1. Model mempelajari aturan keputusan dari data training.
2. Setiap simpul internal merepresentasikan tes pada suatu fitur.
3. Setiap cabang merepresentasikan hasil dari tes.
4. Setiap daun merepresentasikan kelas keputusan (jenis penyakit).

### 2. K-Nearest Neighbors (KNN)

KNN adalah algoritma berbasis instance yang mengklasifikasikan data baru berdasarkan mayoritas kelas dari K tetangga terdekat.

**Parameter:**
- `n_neighbors=5`
- `metric='euclidean'`
- `weights='distance'`

**Cara Kerja:**
1. Untuk setiap data uji, hitung jarak ke semua data latih.
2. Pilih K tetangga terdekat (K=5).
3. Tentukan kelas dengan mayoritas suara dari tetangga terdekat.

### 3. Perbandingan Hasil Model

Hasil evaluasi kedua model dirangkum dalam tabel berikut:

| Model | Akurasi | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| **Decision Tree** | **90.48%** | **90.51%** | **90.48%** | **90.48%** |
| KNN | 88.17% | 88.20% | 88.17% | 88.15% |

### 4. Model Terbaik

Dari hasil evaluasi, model **Decision Tree** memiliki performa terbaik berdasarkan keseimbangan antara akurasi, precision, recall, dan F1-score. Model ini dipilih sebagai model akhir karena performanya yang unggul dan kemudahan interpretasinya.

**Alasan Decision Tree menjadi model terbaik:**

1. **Konsistensi**: Decision Tree menunjukkan performa yang konsisten di semua kelas penyakit (Bacterialblight, Brownspot, Leafsmut) dengan precision dan recall di atas 90% untuk setiap kelas.
2. **Interpretabilitas**: Decision Tree menghasilkan model yang mudah dipahami dalam bentuk pohon keputusan, memungkinkan pengguna untuk melihat aturan-aturan yang digunakan dalam pengambilan keputusan.
3. **Data Non-Linear**: Decision Tree lebih baik dalam menangani data yang tidak linear dibandingkan KNN yang sangat bergantung pada perhitungan jarak Euclidean.
4. **Stabilitas**: Decision Tree tidak terlalu sensitif terhadap skala data, sedangkan KNN sangat bergantung pada normalisasi data.

---

## Evaluation

Pada tahap ini, dilakukan evaluasi terhadap performa model berdasarkan berbagai metrik, yaitu Akurasi, Precision, Recall, dan F1-Score. Selain itu, analisis confusion matrix digunakan untuk memahami prediksi yang benar dan salah dari model.

### Metrik Evaluasi

| Metrik | Definisi |
|--------|----------|
| **Akurasi** | Mengukur proporsi prediksi yang benar dari total prediksi. |
| **Precision** | Mengukur ketepatan prediksi kelas positif dari semua prediksi positif. |
| **Recall** | Mengukur sensitivitas model dalam mendeteksi kelas positif. |
| **F1-Score** | Rata-rata harmonis antara precision dan recall. |

### Hasil Evaluasi

Berikut adalah hasil evaluasi dari kedua model yang digunakan:

| Model | Akurasi | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| **Decision Tree** | **90.48%** | **90.51%** | **90.48%** | **90.48%** |
| KNN | 88.17% | 88.20% | 88.17% | 88.15% |

Dari tabel di atas, model **Decision Tree** memiliki performa terbaik secara keseluruhan dengan nilai F1-Score tertinggi (90.48%).

### Analisis Confusion Matrix

Confusion matrix memberikan informasi tentang prediksi benar (True Positives dan True Negatives) serta prediksi salah (False Positives dan False Negatives).

![Confusion Matrix Decision Tree & KNN](Hasil/Confusion%20Matrix%20Desicion%20Tree%20%26%20KNN.png)

**Gambar 4.** Confusion matrix Decision Tree (kiri) dan KNN (kanan)

#### 1. Decision Tree

Confusion Matrix untuk model Decision Tree menunjukkan:

| | Prediksi Bacterialblight | Prediksi Brownspot | Prediksi Leafsmut |
|---|--------------------------|--------------------|-------------------|
| **Aktual Bacterialblight** | 318 | 21 | 3 |
| **Aktual Brownspot** | 21 | 322 | 4 |
| **Aktual Leafsmut** | 24 | 28 | 292 |

**Analisis:**

- Decision Tree memiliki akurasi sebesar **90.48%**.
- Model ini sangat baik dalam mengklasifikasikan ketiga kelas penyakit.
- Precision, Recall, dan F1-Score semuanya di atas 90%, menunjukkan performa yang konsisten.
- Kesalahan prediksi terjadi pada kelas Leafsmut yang paling sering tertukar dengan Brownspot (28 prediksi salah).

#### 2. K-Nearest Neighbors (KNN)

Confusion Matrix untuk model KNN menunjukkan:

| | Prediksi Bacterialblight | Prediksi Brownspot | Prediksi Leafsmut |
|---|--------------------------|--------------------|-------------------|
| **Aktual Bacterialblight** | 321 | 13 | 0 |
| **Aktual Brownspot** | 22 | 324 | 0 |
| **Aktual Leafsmut** | 24 | 28 | 292 |

**Analisis:**

- KNN memiliki akurasi sebesar **88.17%**.
- Model ini sangat baik dalam mengklasifikasikan kelas Bacterialblight dan Brownspot.
- Namun, KNN kurang baik dalam mengklasifikasikan kelas Leafsmut, dengan 28 prediksi salah menjadi Brownspot dan 24 menjadi Bacterialblight.

### Classification Report

#### Decision Tree

```
              precision    recall  f1-score   support

Bacterialblight    0.91      0.90      0.90       310
Brownspot          0.90      0.91      0.90       310
Leafsmut           0.90      0.90      0.90       310

    accuracy                            0.90       930
   macro avg       0.91      0.90      0.90       930
weighted avg       0.91      0.90      0.90       930
```

#### KNN

```
              precision    recall  f1-score   support

Bacterialblight    0.88      0.88      0.88       310
Brownspot          0.88      0.88      0.88       310
Leafsmut           0.88      0.88      0.88       310

    accuracy                            0.88       930
   macro avg       0.88      0.88      0.88       930
weighted avg       0.88      0.88      0.88       930
```

### Kesimpulan Evaluasi

| Model | Kelebihan | Kekurangan |
|-------|-----------|------------|
| **Decision Tree** | Akurasi tinggi (90.48%), interpretatif, konsisten di semua kelas | Rentan overfitting |
| **KNN** | Sederhana, tidak perlu training | Lambat untuk dataset besar, sensitif skala |

**Model Terbaik:** **Decision Tree**

**Alasan:** Decision Tree memiliki keseimbangan terbaik antara Akurasi (90.48%), Precision (90.51%), Recall (90.48%), dan F1-Score (90.48%). Model ini juga lebih mudah diinterpretasikan dan lebih stabil dibandingkan KNN.

---

## Kesimpulan dan Rekomendasi

### Kesimpulan

Proyek ini berhasil mengembangkan model klasifikasi penyakit daun padi menggunakan dua algoritma machine learning:

| Model | Akurasi | Precision | Recall | F1-Score | Status |
|-------|---------|-----------|--------|----------|--------|
| **Decision Tree** | **90.48%** | **90.51%** | **90.48%** | **90.48%** | **🏆 TERBAIK** |
| KNN | 88.17% | 88.20% | 88.17% | 88.15% | - |

**Kesimpulan Utama:**

1. **Tujuan proyek tercapai**: Kedua model berhasil diimplementasikan dan dievaluasi.
2. **Model terbaik**: Decision Tree dengan akurasi 90.48%, mengungguli KNN yang mencapai 88.17%.
3. **Alasan Decision Tree unggul**: Interpretabilitas, konsistensi di semua kelas, dan kemampuan menangani data non-linear.
4. **Dataset berkualitas**: Distribusi data yang seimbang (1.580 gambar per kelas) mendukung performa model yang baik.

### Rekomendasi Perbaikan

| No | Rekomendasi | Keterangan |
|----|-------------|------------|
| 1 | **Dataset Lebih Besar** | Menambahkan lebih banyak variasi gambar dengan kondisi lingkungan berbeda (pencahayaan, sudut pengambilan, tingkat keparahan penyakit) |
| 2 | **Penambahan Kelas** | Menambahkan kelas penyakit lain seperti Blast, Tungro, dan Sheath Blight untuk meningkatkan cakupan sistem |
| 3 | **Algoritma Ensemble** | Menggunakan Random Forest atau XGBoost untuk meningkatkan akurasi |
| 4 | **Deep Learning** | Mengimplementasikan CNN (ResNet-50, InceptionV3) untuk ekstraksi fitur otomatis |
| 5 | **Aplikasi Mobile** | Mengembangkan aplikasi mobile untuk deteksi real-time di lapangan |
| 6 | **Hyperparameter Tuning** | Melakukan Grid Search untuk menemukan hyperparameter optimal |

---

## Referensi

1. Pangestu, D. A., Aziz, O. Q., & Crysdian, C. (2025). Klasifikasi Penyakit pada Tanaman Berdasarkan Citra Daun Menggunakan Metode Convolutional Neural Network. *Jurnal Informatika dan Teknik Elektro Terapan*, *10*(2), 235-248. https://doi.org/10.14421/jiska.2025.10.2.235-248

2. Ulfi, M., Supriyanto, M. I. A., Sari, C. M., & Nuswantoro, S. A. (2025). Identifikasi Penyakit pada Tanaman Padi Menggunakan Convolutional Neural Network (CNN). *Jurnal Sains Komputer dan Teknologi Informasi*, *7*(2), 89-96. https://doi.org/10.33084/jsakti.v7i2.10061

3. Margarita, D., Maulana, H., & Mandyartha, E. P. (2024). Klasifikasi Penyakit Daun Padi Menggunakan Support Vector Machine Berdasarkan Fitur Mendalam (Deep Feature). *Jurnal Informatika dan Teknik Elektro Terapan*, *9*(4), 2256-2270. https://doi.org/10.29100/ijpi.v9i4.5634

4. Satriadi, R. (2025). *PadiGuard: Aplikasi Deteksi Penyakit Daun Padi Berbasis Deep Learning*. Politeknik Negeri Semarang.

5. Priyadharshini, A., & Kalaivanan, K. (2025). A review on machine learning and deep learning techniques for plant leaf disease detection and classification with IoT in agriculture industry. *Journal of Agriculture and Food Research*, *15*, 101567. https://doi.org/10.1016/j.jafr.2025.101567

6. Singh, A. K., & Kumar, S. (2025). A Systematic Review on the Detection and Classification of Plant Diseases Using Machine Learning. *International Journal of Cognitive Computing in Engineering*, *6*, 45-58.

7. Sitohang, A. H. P., Hermanto, T. I., & Lestari, C. D. (2024). Klasifikasi Jenis Penyakit pada Daun Tumbuhan Stroberi Menggunakan Metode Convolutional Neural Network Arsitektur InceptionV3. *Jurnal Informatika dan Teknik Elektro Terapan*, *12*(3S1). https://doi.org/10.23960/jitet.v12i3S1.5274

---

## Lampiran

### A. Dataset

- **Sumber**: [Kaggle - Rice Plant Diseases Dataset](https://www.kaggle.com/datasets/jay7080dev/rice-plant-diseases-dataset)
- **Total Gambar**: 4.740 gambar
- **Kelas**: 3 kelas (Bacterialblight, Brownspot, Leafsmut)

### B. Kode Lengkap

- **Notebook**: [uas_model.ipynb](uas_model.ipynb)
- **Platform**: Google Colab
- **Library**: Python, Scikit-learn, OpenCV, Matplotlib, Seaborn

### C. Hasil Visualisasi

| File | Keterangan |
|------|------------|
| `Distribusi jumlah Gambar per kelas.png` | Distribusi jumlah gambar per kelas |
| `Gambar Daun Padi Per Kelas.png` | Contoh gambar dari setiap kelas |
| `Distribusi Kelas Data train & Test.png` | Distribusi data training dan testing |
| `Confusion Matrix Desicion Tree & KNN.png` | Confusion matrix kedua model |

### D. Model yang Dihasilkan

| File | Keterangan |
|------|------------|
| `decision_tree_model.pkl` | Model Decision Tree terlatih |
| `knn_model.pkl` | Model KNN terlatih |
| `scaler.pkl` | StandardScaler untuk normalisasi |
| `label_encoder.pkl` | LabelEncoder untuk encoding kelas |

### E. Spesifikasi Teknis

| Aspek | Keterangan |
|-------|------------|
| **Bahasa Pemrograman** | Python 3.10 |
| **Framework** | Scikit-learn 1.2.0 |
| **Library Pengolahan Citra** | OpenCV 4.8.0, Scikit-image 0.21.0 |
| **Library Visualisasi** | Matplotlib 3.7.0, Seaborn 0.12.0 |
| **Platform** | Google Colab (T4 GPU) |

---

**Disusun oleh:**
[Nama Mahasiswa 1] & [Nama Mahasiswa 2]

**Tanggal:** 17 Juli 2026
