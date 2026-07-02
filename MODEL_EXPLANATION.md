# Model Explanation — Stack Overflow AI Tools ML

Dokumen ini menjelaskan alasan pemilihan dan cara kerja setiap model yang digunakan dalam proyek ini.

---

## Daftar Model

| # | Model | Kategori | Tipe | Tujuan |
|---|-------|----------|------|--------|
| 1 | **K-Means** | Tradisional | Unsupervised | Segmentasi developer |
| 2 | **GMM** | Modern | Unsupervised | Segmentasi developer (probabilistic) |
| 3 | **Random Forest** | Tradisional | Supervised | Klasifikasi AI Usage |
| 4 | **Linear SVM** | Tradisional | Supervised | Klasifikasi AI Usage |
| 5 | **XGBoost** | Modern | Supervised | Klasifikasi AI Usage |
| 6 | **MLP (Neural Network)** | Modern | Supervised | Klasifikasi AI Usage |

---

## A. Unsupervised Learning — Segmentasi Developer

### 1. K-Means (Traditional)

**Alasan Pemilihan:**
- Algoritma clustering paling populer dan mudah diinterpretasi.
- Cocok untuk segmentasi awal karena sederhana dan cepat.
- Menjadi **baseline** untuk perbandingan dengan metode clustering modern.

**Cara Kerja:**
1. Inisialisasi `k` centroid secara acak.
2. Setiap data point ditetapkan ke centroid terdekat (jarak Euclidean).
3. Centroid dihitung ulang sebagai rata-rata semua point dalam cluster.
4. Langkah 2-3 diulang sampai centroid tidak berubah (konvergen).

**Parameter:**
- `k`: dipilih berdasarkan Silhouette Score tertinggi (diuji 2–10).
- `n_init=10`: inisialisasi ulang 10 kali untuk menghindari local optimum.
- `random_state=42`: reproducible.

**Kelemahan pada Dataset Ini:**
- Hanya menghasilkan **k=2** (minimum), artinya data tidak memiliki struktur cluster yang tegas.
- Rentan terhadap curse of dimensionality — itulah mengapa data direduksi dengan TruncatedSVD sebelum clustering.

---

### 2. GMM — Gaussian Mixture Model (Modern)

**Alasan Pemilihan:**
- Lebih fleksibel dari K-Means karena cluster bisa berbeda bentuk dan ukuran.
- Memberikan **soft clustering**: setiap developer punya probabilitas masuk ke tiap cluster.
- Modern probabilistic approach — cocok dibandingkan dengan K-Means yang rigid.

**Cara Kerja:**
1. Asumsi data berasal dari campuran beberapa distribusi Gaussian.
2. **Expectation-Maximization (EM)** digunakan untuk memperkirakan parameter tiap Gaussian (mean, covariance, weight).
3. E-step: hitung probabilitas tiap point terhadap tiap cluster.
4. M-step: update parameter distribusi berdasarkan probabilitas tersebut.
5. Ulang sampai konvergen.

**Parameter:**
- `k`: sama seperti K-Means, dipilih berdasarkan Silhouette Score (diuji 2–8).
- `n_init=3`: inisialisasi ulang 3 kali (dikurangi dari default 5 untuk efisiensi).
- `covariance_type='full'`: tiap cluster punya covariance matrix sendiri.

**Kelebihan pada Dataset Ini:**
- Menghasilkan **k=8** (lebih informatif dari K-Means yang hanya k=2).
- Probabilitas cluster dapat digunakan untuk analisis lebih lanjut.
- Cluster bisa memiliki bentuk ellipsoid, tidak hanya spherical seperti K-Means.

---

## B. Supervised Learning — Klasifikasi AI Usage

**Target:** `AI_Usage` (1 = menggunakan AI tools, 0 = tidak).

**Imbalance:** 78.5% kelas 1 vs 21.5% kelas 0. Semua model menggunakan `class_weight="balanced"` atau `scale_pos_weight` untuk mengatasi imbalance.

### 3. Random Forest (Traditional)

**Alasan Pemilihan:**
- Ensemble method yang robust terhadap outlier dan overfitting.
- Menangani campuran fitur numerik, kategorikal, dan multi-select dengan baik.
- Memberikan feature importance untuk interpretasi.

**Cara Kerja:**
1. Membangun banyak decision tree (200 trees) pada bootstrap sample data.
2. Setiap tree hanya mempertimbangkan subset fitur acak saat split.
3. Prediksi akhir: voting mayoritas dari semua tree.

**Parameter:**
- `n_estimators=200`: jumlah pohon.
- `class_weight="balanced"`: menangani imbalance.
- `n_jobs=-1`: menggunakan semua CPU core.

**Hasil:** F1-score = **0.867** (kelas AI User) — model kedua terbaik setelah MLP.

---

### 4. Linear SVM / LinearSVC (Traditional)

**Alasan Pemilihan:**
- Model linear yang sederhana dan mudah diinterpretasi.
- Sebagai **baseline** untuk perbandingan dengan model non-linear (Random Forest, XGBoost, MLP).
- Bekerja baik pada data high-dimensional sparse (hasil OHE + CountVectorizer).

**Cara Kerja:**
1. Mencari hyperplane (decision boundary) yang memisahkan dua kelas dengan margin maksimal.
2. LinearSVC menggunakan loss function hinge loss dengan regularisasi L2.
3. Prediksi berdasarkan sisi hyperplane tempat data point berada.

**Parameter:**
- `class_weight="balanced"`: mengatasi imbalance.
- `max_iter=5000`: iterasi maksimal untuk konvergensi.
- `C=1.0`: regularization strength (default).

**Kelemahan pada Dataset Ini:**
- Decision boundary linear tidak cukup untuk memisahkan pengguna vs non-pengguna AI.
- Performa terendah: F1 = **0.760**.
- Karena hubungan antara karakteristik developer dan penggunaan AI bersifat **non-linear**.

---

### 5. XGBoost (Modern)

**Alasan Pemilihan:**
- State-of-the-art untuk tabular data — sering menang di kompetisi Kaggle.
- Gradient boosting yang lebih optimal dari Random Forest.
- Handle missing value secara native.

**Cara Kerja:**
1. Membangun pohon secara **sequential**: setiap pohon baru memperbaiki error pohon sebelumnya.
2. Menggunakan **gradient descent** untuk meminimalkan loss function.
3. Regularisasi (L1/L2) untuk mencegah overfitting.

**Parameter:**
- `n_estimators=200`: jumlah pohon boosting.
- `learning_rate=0.1`: shrinkage untuk mencegah overfitting.
- `max_depth=6`: kedalaman pohon.
- `scale_pos_weight`: rasio negatif/positif untuk mengatasi imbalance.
- `eval_metric='logloss'`: metrik evaluasi selama training.

**Hasil:** F1-score = **0.789** — lebih rendah dari Random Forest. Kemungkinan karena tuning hyperparameter belum optimal (masih default baseline).

---

### 6. MLP — Multi-Layer Perceptron (Modern)

**Alasan Pemilihan:**
- Neural network untuk tabular data — menangkap pola non-linear yang kompleks.
- Satu-satunya model deep learning dalam proyek ini.
- Perbandingan menarik: apakah neural network mengalahkan tree-based model?

**Cara Kerja:**
1. Input layer: fitur hasil preprocessing (dense representation).
2. Hidden layer 1: 100 neuron dengan aktivasi ReLU.
3. Hidden layer 2: 50 neuron dengan aktivasi ReLU.
4. Output layer: 1 neuron dengan sigmoid (binary classification).
5. Backpropagation dengan Adam optimizer untuk meminimalkan binary cross-entropy.
6. **Early stopping** untuk mencegah overfitting.

**Parameter:**
- `hidden_layer_sizes=(100, 50)`: dua hidden layer.
- `max_iter=500`: maksimal epoch.
- `early_stopping=True`: stop jika validation loss tidak membaik.
- `random_state=42`: reproducible.

**Hasil:** F1-score = **0.882** — **model terbaik**. Menarik karena neural network outperform ensemble methods pada dataset ini.

---

## C. Perbandingan Traditional vs Modern

### Unsupervised

| Aspek | K-Means (Traditional) | GMM (Modern) |
|-------|----------------------|--------------|
| Jenis cluster | Hard (tegas) | Soft (probabilistik) |
| Bentuk cluster | Spherical saja | Ellipsoid (bervariasi) |
| k terbaik | 2 | 8 |
| Interpretasi | Mudah | Sedang |

### Supervised

| Aspek | RF & SVM (Traditional) | XGBoost & MLP (Modern) |
|-------|----------------------|----------------------|
| Kompleksitas | Rendah–Sedang | Sedang–Tinggi |
| Training time | Cepat | Sedang–Lama |
| Performa terbaik | RF: 0.867 | **MLP: 0.882** |
| Interpretasi | RF: feature importance tersedia | Sulit (black box) |
| Overfitting risk | Rendah (RF ensemble) | Sedang (bisa diatasi early stopping) |

---

## D. Dataset Considerations

**Mengapa model tertentu cocok/tidak cocok:**

1. **Imbalance (78:22)** → Semua supervised model pakai class weighting, tapi tetap kesulitan memprediksi kelas minoritas (non-AI user). Lihat tabel classification report: recall Non-AI hanya 0.17–0.68.

2. **Multi-select features** (Language, Database, Platform, Webframe) → Diencode sebagai sparse binary vectors via CountVectorizer. Tree-based models (RF, XGBoost) dan SVM menangani sparse data dengan baik. MLP juga bisa karena hidden layer belajar dense representation.

3. **Missing YearsCodePro & ToolsTechHaveWorkedWith** → Dua fitur penting tidak tersedia, yang mungkin berkontribusi pada performa model yang belum optimal.

4. **High cardinality** (Country: 100+ unique values, DevType: banyak kombinasi) → One-Hot Encoding menghasilkan dimensi tinggi. Namun TruncatedSVD untuk clustering dan tree-based models untuk klasifikasi dapat menanganinya.

---

## E. Kesimpulan untuk BAB IV

Dari 6 model yang diuji:

- **K-Means (k=2)**: segmentasi dasar, interpretasi mudah, tapi cluster kurang informatif.
- **GMM (k=8)**: segmentasi lebih granular, cocok untuk analisis segmentasi yang lebih dalam.
- **Random Forest**: baseline klasifikasi yang solid (F1=0.867).
- **Linear SVM**: paling lemah, menunjukkan hubungan non-linear dalam data.
- **XGBoost**: potensi besar dengan tuning lebih lanjut.
- **MLP Neural Network**: **terbaik** (F1=0.882), membuktikan neural network efektif untuk tabular data developer.

Rekomendasi untuk deployment: **Random Forest** jika interpretability penting, **MLP** jika performa murni yang diutamakan.
