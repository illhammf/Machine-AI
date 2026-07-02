# Model Explanation — Stack Overflow AI Tools ML

Dokumen ini menjelaskan struktur model, alasan pemilihan, dan cara kerja setiap algoritma yang digunakan dalam proyek Machine Learning berbasis Stack Overflow Developer Survey 2025.

Tujuan utama proyek ini adalah:
1. Mengelompokkan developer berdasarkan karakteristik mereka (unsupervised learning)
2. Memprediksi penggunaan AI tools oleh developer (supervised learning)
3. Membandingkan pendekatan tradisional dan modern dalam Machine Learning

---

# 1. Daftar Model yang Digunakan

| No | Model | Jenis | Tipe Learning | Tujuan |
|----|------|------|--------------|--------|
| 1 | K-Means | Traditional | Unsupervised | Segmentasi developer |
| 2 | Gaussian Mixture Model (GMM) | Modern | Unsupervised | Segmentasi probabilistik developer |
| 3 | Random Forest | Traditional | Supervised | Klasifikasi penggunaan AI |
| 4 | Linear SVM | Traditional | Supervised | Klasifikasi baseline |
| 5 | XGBoost | Modern | Supervised | Klasifikasi performa tinggi |
| 6 | MLP Neural Network | Modern | Supervised | Klasifikasi non-linear kompleks |

---

# 2. Unsupervised Learning (Segmentasi Developer)

Tujuan tahap ini adalah mengelompokkan developer berdasarkan pola perilaku, teknologi, dan pengalaman tanpa label target.

---

## 2.1 K-Means Clustering (Baseline Model)

### Tujuan
K-Means digunakan sebagai model baseline karena sederhana, cepat, dan mudah diinterpretasikan.

### Cara Kerja
1. Menentukan jumlah cluster (k)
2. Menentukan centroid secara acak
3. Menghitung jarak setiap data ke centroid (Euclidean distance)
4. Mengelompokkan data ke centroid terdekat
5. Menghitung ulang centroid berdasarkan rata-rata cluster
6. Mengulangi proses hingga stabil (konvergen)

### Parameter
- k dipilih berdasarkan Silhouette Score (range 2–10)
- n_init = 10 untuk menghindari hasil lokal optimum
- random_state = 42 untuk reproduktibilitas

### Kelebihan
- Cepat dan efisien
- Mudah dipahami

### Keterbatasan
- Hanya mampu membentuk cluster berbentuk bulat (spherical)
- Pada dataset ini hanya menghasilkan k kecil (struktur data tidak terlalu kuat)

---

## 2.2 Gaussian Mixture Model (GMM)

### Tujuan
GMM digunakan untuk meningkatkan fleksibilitas clustering dengan pendekatan probabilistik.

### Cara Kerja
GMM mengasumsikan data berasal dari beberapa distribusi Gaussian.

Proses utama:
1. Expectation Step: menghitung probabilitas data terhadap setiap cluster
2. Maximization Step: memperbarui parameter distribusi (mean, covariance, weight)
3. Iterasi hingga konvergen

### Parameter
- k dipilih menggunakan Silhouette Score (2–8)
- covariance_type = "full"
- n_init = 3

### Kelebihan
- Mendukung soft clustering (probabilitas ke setiap cluster)
- Cluster dapat berbentuk ellipsoid, tidak terbatas bentuk bulat
- Lebih realistis untuk data kompleks

### Hasil
- Menghasilkan cluster lebih banyak dibanding K-Means
- Lebih informatif untuk analisis segmentasi

---

# 3. Supervised Learning (Klasifikasi AI Usage)

## Target Variabel
AI_Usage:
- 1 = menggunakan AI tools
- 0 = tidak menggunakan AI tools

Dataset memiliki ketidakseimbangan kelas:
- 78.5% kelas mayoritas
- 21.5% kelas minoritas

Semua model menggunakan teknik balancing (class_weight atau scale_pos_weight).

---

## 3.1 Random Forest

### Tujuan
Model ensemble sebagai baseline kuat untuk klasifikasi tabular data.

### Cara Kerja
1. Membuat banyak decision tree dari bootstrap sample
2. Setiap tree memilih subset fitur secara acak
3. Hasil akhir ditentukan melalui voting mayoritas

### Parameter
- n_estimators = 200
- class_weight = balanced
- n_jobs = -1

### Kelebihan
- Stabil terhadap noise dan outlier
- Tidak mudah overfitting
- Dapat menampilkan feature importance

### Hasil
F1-score: 0.867

---

## 3.2 Linear SVM

### Tujuan
Digunakan sebagai baseline model linear.

### Cara Kerja
SVM mencari hyperplane terbaik yang memaksimalkan margin antar kelas.

### Parameter
- C = 1.0
- max_iter = 5000
- class_weight = balanced

### Kelemahan
- Tidak mampu menangkap hubungan non-linear
- Performa paling rendah di antara semua model

### Hasil
F1-score: 0.760

---

## 3.3 XGBoost

### Tujuan
Model boosting modern untuk meningkatkan akurasi prediksi.

### Cara Kerja
1. Model dibangun secara sequential
2. Setiap model baru memperbaiki error model sebelumnya
3. Menggunakan gradient descent untuk optimasi loss function

### Parameter
- n_estimators = 200
- learning_rate = 0.1
- max_depth = 6
- scale_pos_weight untuk imbalance

### Kelebihan
- Sangat kuat untuk data tabular
- Menangani missing value secara otomatis
- Digunakan luas dalam industri dan kompetisi data science

### Hasil
F1-score: 0.789

---

## 3.4 MLP Neural Network

### Tujuan
Menguji kemampuan model deep learning pada data tabular.

### Cara Kerja
1. Input layer menerima fitur hasil preprocessing
2. Hidden layer (100 neuron → 50 neuron)
3. Aktivasi ReLU
4. Output layer sigmoid untuk klasifikasi biner
5. Optimasi menggunakan Adam optimizer
6. Early stopping untuk mencegah overfitting

### Parameter
- hidden_layer_sizes = (100, 50)
- max_iter = 500
- early_stopping = True

### Kelebihan
- Mampu menangkap pola non-linear kompleks
- Fleksibel terhadap data besar

### Hasil
F1-score: 0.882 (terbaik)

---

# 4. Perbandingan Model

## 4.1 Unsupervised Learning

| Model | Tipe Cluster | Interpretasi | Kelebihan |
|------|-------------|--------------|------------|
| K-Means | Hard clustering | Mudah | Cepat dan sederhana |
| GMM | Soft clustering | Menengah | Lebih fleksibel |

---

## 4.2 Supervised Learning

| Model | Tipe | Performa | Kelebihan |
|------|------|---------|------------|
| SVM | Linear | Rendah | Sederhana |
| Random Forest | Ensemble | Tinggi | Stabil dan interpretatif |
| XGBoost | Boosting | Menengah-Tinggi | Sangat kuat untuk tabular |
| MLP | Neural Network | Tertinggi | Non-linear kompleks |

---

# 5. Analisis Dataset

Dataset memiliki karakteristik:

1. Imbalance data (78:22)
   → Menyulitkan prediksi kelas minoritas

2. Fitur multi-label (Language, Platform, Tools)
   → Perlu encoding khusus (CountVectorizer / OneHot)

3. Data kategorikal dominan
   → Cocok untuk tree-based models

4. Dimensi tinggi
   → Menggunakan reduksi dimensi untuk clustering

---

# 6. Kesimpulan

- K-Means dan GMM berhasil memberikan gambaran segmentasi developer
- Semua model supervised berhasil memprediksi AI usage dengan performa berbeda
- Model terbaik adalah MLP dengan F1-score tertinggi (0.882)
- Dataset menunjukkan pola non-linear sehingga model modern lebih unggul

---

# 7. Rekomendasi

- Gunakan Random Forest jika interpretasi penting
- Gunakan MLP jika fokus pada performa
- Gunakan GMM untuk analisis segmentasi probabilistik
- Gunakan XGBoost untuk eksperimen tuning lanjutan