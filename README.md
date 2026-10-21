# fsd_linear_kelompok3
Tugas Kelompok: Linear Regression dan Logistic Regression

Anggota Kelompok: 
- Laksita Ayudia Nurislami (25523182)
- Fathin Nishrina Nurul Aulya (25523256)
- Fina Nailatul 'Izzah (25523264)
  
## Pembagian Kontribusi

| Anggota | Kontribusi |
|---|---|
| Laksita Ayudia Nurislami (25523182) | Bagian B (B1–B8): pengerjaan Logistic Regression |
| Fathin Nishrina Nurul Aulya (25523256) | Bagian A, tahap A1–A4: deskripsi dataset, eksplorasi dan visualisasi, preprocessing, dan pemilihan fitur Linear Regression |
| Fina Nailatul 'Izzah (25523264) | Bagian A, tahap A5–A8: pembagian data, pemodelan, evaluasi, dan analisis performa Linear Regression |---|---|
| Seluruh anggota | Penyusunan README |

# Bagian A: Linear Regression
## A1. Deskripsi Dataset

Sumber dataset: California Housing (`california_housing_train.csv`), tersedia di folder `sample_data` Google Colab. Dataset ini berisi data sensus California tahun 1990, dengan satu baris mewakili satu blok wilayah (*block group*).

Ukuran dataset: terdiri dari 17.000 baris dan 9 kolom (8 fitur + 1 target)

| Kolom | Tipe | Arti |
|---|---|---|
| `longitude` | Fitur | Posisi bujur blok (makin besar, makin ke timur) |
| `latitude` | Fitur | Posisi lintang blok (makin besar, makin ke utara) |
| `housing_median_age` | Fitur | Median usia rumah di blok (tahun) |
| `total_rooms` | Fitur | Total jumlah ruangan di blok |
| `total_bedrooms` | Fitur | Total jumlah kamar tidur di blok |
| `population` | Fitur | Jumlah penduduk di blok |
| `households` | Fitur | Jumlah rumah tangga di blok |
| `median_income` | Fitur | Median pendapatan rumah tangga di blok (satuan puluhan ribu USD) |
| `median_house_value` | **Target** | Median harga rumah di blok (USD) |

Tujuan: memprediksi `median_house_value` berdasarkan 8 fitur di atas.

## A2. Eksplorasi dan Visualisasi Data

Pada tahap ini kita melihat statistik deskriptif dan membuat tiga visualisasi untuk memahami distribusi target, hubungan antarfitur, serta hubungan fitur dengan target.

**Statistik deskriptif:** dataset berisi 17.000 baris dengan 8 fitur dan 1 target, semuanya numerik. Setiap kolom terisi penuh (count 17.000), jadi tidak ada indikasi missing value. Target `median_house_value` memiliki rata-rata sekitar 207.301 USD dan median 180.400 USD (rentang 14.999 sampai 500.001 USD). Fitur `total_rooms`, `total_bedrooms`, `population`, dan `households` memiliki nilai maksimum jauh di atas kuartil ketiga, yang menandakan outlier.

**Tiga visualisasi dan temuannya:**

| Visualisasi | Temuan utama |
|---|---|
| Histogram target | Distribusi condong ke kanan, dengan sebagian besar blok berharga 100.000 sampai 250.000 USD. Ada lonjakan di 500.000 USD karena harga dibatasi (*capped*) di 500.001, bukan pola alami. |
| Heatmap korelasi | `median_income` berkorelasi paling kuat dengan target (r = 0,69), sedangkan fitur lain lemah (|r| ≤ 0,14). `total_rooms`, `total_bedrooms`, `population`, dan `households` saling berkorelasi sangat tinggi (0,86 sampai 0,98), sehingga berpotensi multikolinearitas. `longitude` dan `latitude` berkorelasi -0,93. |
| Scatter plot fitur vs target | `median_income` menunjukkan tren naik yang jelas, dengan sebaran yang melebar di pendapatan tinggi. `housing_median_age` dan `total_rooms` tidak menunjukkan pola linear. Garis horizontal di 500.000 terlihat di semua plot (efek *capping*), dan `total_rooms` memiliki titik-titik ekstrem di sisi kanan. |

**Implikasi untuk tahap berikutnya:** nilai target yang di-*cap* dan outlier fitur perlu ditangani di A3, sedangkan multikolinearitas antar fitur ukuran blok dibahas di A4.

## A3. Preprocessing
Langkah (1)Cek Missing Values (2) Cek Data Duplikat (3) Cek Outlier
### Keputusan preprocessing (A3)

| Langkah | Hasil pengecekan | Keputusan dan alasan |
|---|---|---|
| Missing values | Tidak ada (0 di semua 9 kolom) | Tidak perlu imputasi atau penghapusan baris. |
| Data duplikat | 0 baris duplikat | Tidak ada baris yang dihapus. `drop_duplicates()` tetap dijalankan sebagai pengaman. |
| Target *capped* | 814 baris (4,8%) bernilai 500.001 | **Dihapus.** Nilai ini batas atas data (*censored*), bukan harga sebenarnya. Jika dipertahankan, garis regresi tertarik ke nilai yang tidak nyata. |
| Outlier fitur (`total_rooms`, `total_bedrooms`, `population`, `households`) | Sekitar 1.015 sampai 1.076 outlier per fitur (sekitar 6% data), dideteksi dengan IQR | **Di-*clip* ke batas IQR**, tidak dihapus. Menghapusnya membuang banyak baris, sedangkan outlier ini kemungkinan blok berpenduduk padat yang valid, bukan kesalahan data. |
| Encoding | Semua fitur numerik | Tidak diperlukan. |
| Scaling | - | Tidak diperlukan untuk prediksi OLS. `StandardScaler` hanya dipakai setelah pembagian data (fit pada data latih saja, mencegah *data leakage*) untuk membandingkan koefisien antar fitur. |

Ukuran data akhir setelah preprocessing: **16.186 baris x 9 kolom** (17.000 dikurangi 814 baris target *capped*).

## A4. Pemilihan Fitur
Fitur dipilih dengan tiga analisis: (1) korelasi dengan target, (2) multikolinearitas dengan VIF, (3) skor F dari `SelectKBest`.

| Fitur | Korelasi dgn target | VIF | Skor F | Keputusan |
|---|---|---|---|---|
| `median_income` | 0,65 | 2,1 | 11.574 | Dipertahankan |
| `households` | 0,11 | 33,1 | 203 | Dipertahankan (wakil ukuran blok) |
| `housing_median_age` | 0,07 | 1,3 | 75 | Dipertahankan |
| `latitude` | -0,15 | 9,3 | 375 | Dipertahankan (efek lokasi) |
| `longitude` | -0,04 | 9,0 | 31 | Dipertahankan (efek lokasi) |
| `total_rooms` | 0,18 | 13,2 | 537 | Dibuang (redundan) |
| `total_bedrooms` | 0,09 | 32,9 | 129 | Dibuang (redundan) |
| `population` | 0,01 | 6,1 | 1,2 | Dibuang (redundan, skor F terendah) |

`median_income` konsisten menjadi fitur terkuat pada korelasi maupun skor F. Empat fitur ukuran blok saling tumpang tindih, sehingga hanya `households` yang dipertahankan. `latitude` dan `longitude` tetap dipakai karena lokasi sangat memengaruhi harga, dengan catatan koefisiennya tidak diinterpretasikan terpisah. Fitur akhir: 5 fitur, data 16.186 baris.

## A5. Pembagian Data
dengan tujuan utama mengevaluasi kemampuan geleralisasi model secara asil pada data baru yang belum pernah dilihat untuk mencegah overfitting & data leakage, menguji kemampuan prediksi nyata, mendeteksi masalah pemodelan

### 1. Ringkasan Pembagian
Data dibagi menggunakan `train_test_split` dari `scikit-learn` dengan rasio **80:20** dan `random_state=42`.

* **Total Sampel**: 16.186 baris
* **Data Latih (Train Set)**: 12.948 sampel (80%)
* **Data Uji (Test Set)**: 3.238 sampel (20%)
* **Fitur ($X$)**: `longitude`, `latitude`, `housing_median_age`, `households`, `median_income`
* **Target ($y$)**: `median_house_value`

---

### 2. Analisis Pembagian Data

* **Rasio 80:20**: Proporsi ini ideal untuk menjaga kecukupan data saat melatih koefisien regresi sekaligus menyediakan sampel uji yang valid.
* **Penggunaan `random_state=42`**: Mengunci sifat pengacakan (*seed*) agar pembagian data bersifat konsisten dan dapat direproduksi ulang (*reproducible*).
* **Metode Sampling**: Menggunakan *Simple Random Sampling* yang sesuai untuk data numerik kontinu berukuran moderat.
* **Pencegahan Data Leakage**: Pembagian dilakukan sebelum pemrosesan *StandardScaler* agar informasi dari data uji tidak membocorkan statistik ke model latih.

## A6. Pemodelan Linear Regression
Dengan tujuan utama membangun dan mempelajari hubungan matematis antara fitur-fitur pilihan dengan harga rumah seperti mencari model prediksi, mengstandarisasi dan untuk dasar analisis serta evaluasi yang akan digunakan untuk A7. Model dilatih pada data latih. Koefisien ditampilkan dalam satuan asli agar mudah diinterpretasikan, dan dalam bentuk standardisasi agar bisa dibandingkan antar fitur.

## A6. Pemodelan Linear Regression

### 1. Ringkasan Parameter Model
Model **Linear Regression** dilatih menggunakan 12.948 sampel data latih ($X_{train}$ dan $y_{train}$) berbasis 5 fitur terpilih.

* **Nilai Intercept (Satuan Asli)**: `-3.410.403,96`
* **Persamaan Model (Satuan Asli)**:
  $$y = -3.410.403,96 - 40.153,47(\text{longitude}) - 38.872,09(\text{latitude}) + 1.117,60(\text{housing\_median\_age}) + 40,39(\text{households}) + 37.220,75(\text{median\_income})$$

---

### 2. Koefisien & Bobot Pengaruh Fitur

Untuk membandingkan tingkat pengaruh (*importance*) antar fitur tanpa terpengaruh perbedaan satuan, digunakan **Koefisien Terstandardisasi** via `StandardScaler`:

| Urutan | Fitur | Koefisien Asli | Koefisien Terstandardisasi | Arah & Tingkat Pengaruh |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `latitude` | `-38.872,09` | **`-83.432,83`** | Sangat Kuat (Negatif) |
| **2** | `longitude` | `-40.153,47` | **`-80.536,30`** | Sangat Kuat (Negatif) |
| **3** | `median_income` | `+37.220,75` | **`+58.967,83`** | Sangat Kuat (Positif) |
| **4** | `housing_median_age` | `+1.117,60` | **`+14.040,94`** | Moderat (Positif) |
| **5** | `households` | `+40,39` | **`+10.766,94`** | Lemah (Positif) |

---

### 3. Analisis Hasil Temuan A6

* **Pengaruh Posisi Geografis (`latitude` & `longitude`)**:
  Koefisien terstandardisasi menunjukkan bahwa koordinat lokasi memiliki pengaruh absolut terbesar terhadap harga rumah. Nilai negatif yang besar mencerminkan tren geografis wilayah California, di mana perpindahan ke arah timur/utara (menjauhi pesisir utama/area metropolitan tertentu) berkorelasi dengan penurunan harga rumah.
* **Pengaruh Utama Pendapatan (`median_income`)**:
  Merupakan variabel prediktor non-lokasi terkuat. Setiap kenaikan $10.000 pada rata-rata pendapatan rumah tangga berkorelasi dengan kenaikan harga rumah sebesar ±**$37.220,75** (dengan asumsi fitur lain konstan).
* **Efek Usia Bangunan & Ukuran Blok**:
  Usia bangunan (`housing_median_age`) dan jumlah rumah tangga (`households`) memberikan kontribusi positif namun lebih kecil terhadap kenaikan harga rumah.
* **Interpretasi Intercept**:
  Nilai *intercept* negatif yang ekstrem (`-3.410.403,96`) muncul karena tidak ada titik data riil di mana koordinat geografis dan variabel lain bernilai nol. Nilai ini bertindak secara matematis sebagai konstanta penyeimbang dalam skala garis regresi.

## A7. Evaluasi Model

### 1. Tujuan Evaluasi Model
Proses evaluasi model dilakukan untuk:
* **Mengukur Akurasi Prediksi**: Menilai seberapa kecil tingkat kesalahan (*error*) antara harga rumah hasil prediksi model dengan harga aktual.
* **Mendeteksi Overfitting / Underfitting**: Membandingkan metrik performa pada data latih (*train set*) dan data uji (*test set*) untuk memastikan model dapat melakukan generalisasi dengan baik pada data baru.
* **Validasi Asumsi Regresi**: Memeriksa kelayakan model linear melalui analisis distribusi *error* (residual plot).

---

### 2. Metrik Evaluasi

Eksperimen menggunakan 4 metrik evaluasi utama pada kedua himpunan data:

| Metrik Evaluasi | Data Latih (*Train*) | Data Uji (*Test*) | Selisih / Evaluasi |
| :--- | :---: | :---: | :---: |
| **MAE (Mean Absolute Error)** | `48.383,62` | `47.415,29` | -$968,33 |
| **MSE (Mean Squared Error)** | `4,13 x 10⁹` | `4,05 x 10⁹` | -$8,00 x 10⁷ |
| **RMSE (Root Mean Squared Error)** | `64.282,22` | `63.689,57` | -$592,65 |
| **R² Score (Koefisien Determinasi)** | `0,5722` (57,22%) | `0,5671` (56,71%) | -0,0051 (-0,51%) |

---

### 3. Analisis Hasil Evaluasi A7

* **Akurasi & Deviasi Prediksi**:
  * **MAE ($47.415,29)**: Secara rata-rata, prediksi harga rumah meleset sekitar **$47.415** dari harga asli secara absolut.
  * **RMSE ($63.689,57)**: Bobot kesalahan yang lebih besar (penalti untuk *error* ekstrem) menunjukkan angka rata-rata deviasi kuadrat sebesar **$63.689**.
* **Daya Jelaskan Model (R² Score)**:
  * Model mampu menjelaskan **56,71%** variasi harga rumah pada data uji (*unseen data*). Sisa **43,29%** variasi dipengaruhi oleh faktor lain di luar 5 fitur yang digunakan.
* **Keseimbangan Performa (Bebas Overfitting)**:
  * Performa pada *train set* (R²: 0.5722) dan *test set* (R²: 0.5671) sangat identik dengan beda tipis (~0,51%). Hal ini membuktikan bahwa **model tidak mengalami overfitting** dan memiliki kemampuan generalisasi yang stabil.
 
## A8. Analisis Performa

### 1. Kemampuan Prediksi Model
* **R² Score Data Uji ($0,5671$)**: Model mampu menjelaskan **56,71%** variasi harga rumah pada data uji (*unseen data*).
* **Rata-rata Kesalahan (RMSE & MAE)**:
  * **RMSE ($63.689,57)**: Secara rata-rata, estimasi prediksi model memiliki deviasi kesalahan sebesar **$63.689,57** dari harga aktual.
  * **MAE ($47.415,29)**: Selisih mutlak antara harga asli dan harga prediksi rata-rata sebesar **$47.415,29**.

---

### 2. Diagnosis Fitting (Overfitting / Underfitting)
* **Keseimbangan Performa**: Hasil metrik pada data latih ($R^2 = 0,5722$; $\text{RMSE} = 64.282,22$) dan data uji ($R^2 = 0,5671$; $\text{RMSE} = 63.689,57$) menunjukkan nilai yang sangat konsisten.
* **Evaluasi**: Model **tidak mengalami overfitting** (tidak sekadar menghafal data latih). Namun, model berada pada kondisi **underfitting ringan** karena asumsi hubungan linear yang sederhana belum mampu menangkap pola hubungan kompleks pada data real estat.

---

### 3. Hirarki Fitur Paling Berpengaruh
Berdasarkan nilai **koefisien terstandardisasi**, berikut tingkat pengaruh variabel terhadap prediksi target:
1. `latitude` (`-83.432,83`) — Pengaruh negatif terbesar (posisi utara-selatan).
2. `longitude` (`-80.536,30`) — Pengaruh negatif terbesar kedua (posisi barat-timur).
3. `median_income` (`+58.967,83`) — Prediktor positif utama (tingkat pendapatan).
4. `housing_median_age` (`+14.040,94`) — Pengaruh positif sedang (usia bangunan).
5. `households` (`+10.766,94`) — Pengaruh positif paling kecil (skala area).

---

### 4. Keterbatasan Model & Evaluasi Residual
1. **Asumsi Linearitas**: Hubungan antara karakteristik wilayah dan harga properti di lapangan bersifat non-linear, sehingga regresi linear memiliki batas kemampuan penyesuaian (*underfitting*).
2. **Pengaruh Batas *Capping* Target**: Adanya nilai *censored* ($500.001) pada dataset asli membatasi akurasi prediksi untuk rumah berharga sangat tinggi.
3. **Penyederhanaan Geografis**: Penggunaan `latitude` dan `longitude` sebagai nilai linear tunggal kurang optimal dibanding memodelkan variabel spasial non-linear (seperti jarak ke garis pantai atau pusat kota).
4. **Heteroskedastisitas**: Pada *residual plot*, sebaran *error* cenderung membesar seiring meningkatnya nilai harga prediksi, yang melanggar asumsi varians konstan (*homoscedasticity*) pada regresi linear.

# Bagian B: Logistic Regression

## B1. Deskripsi Dataset

**Sumber:** Pokemon with stats (abcsds), Kaggle. https://www.kaggle.com/datasets/abcsds/pokemon

**Ukuran:** 800 baris dan 13 kolom.

**Tujuan:** memprediksi apakah sebuah Pokémon termasuk kategori *Legendary* berdasarkan statistik dasarnya.

**Arti kolom:**
| Kolom | Arti |
|---|---|
| `#` | Nomor Pokédex |
| `Name` | Nama Pokémon |
| `Type 1` | Tipe utama (misalnya Grass, Fire) |
| `Type 2` | Tipe kedua (boleh kosong) |
| `Total` | Jumlah seluruh statistik dasar |
| `HP` | Poin kesehatan |
| `Attack` | Kekuatan serangan fisik |
| `Defense` | Pertahanan fisik |
| `Sp. Atk` | Serangan khusus |
| `Sp. Def` | Pertahanan khusus |
| `Speed` | Kecepatan |
| `Generation` | Generasi kemunculan (1 sampai 6) |
| `Legendary` | Status Legendary (True/False) |

**Variabel target:** `Legendary`
- Kelas 0 (False): bukan Pokémon Legendary
- Kelas 1 (True): Pokémon Legendary (kelas positif)

## B2. Eksplorasi dan Visualisasi Data

**Statistik deskriptif:** dataset berisi 800 baris. Semua kolom numerik terisi penuh (count 800), sedangkan kolom `Type 2` hanya terisi 414 baris. Rata-rata `Total` sekitar 435 dengan rentang 180 sampai 780. Beberapa statistik memiliki nilai maksimum jauh di atas kuartil ketiga (misalnya HP maksimum 255, padahal 75% data di bawah 80), yang menandakan outlier. `Generation` hanya bernilai 1 sampai 6, sehingga lebih tepat dianggap kategori.

**Tiga visualisasi dan temuannya:**

| Visualisasi | Temuan utama |
|---|---|
| Bar chart distribusi kelas | Kelas sangat tidak seimbang: 735 non-Legendary (91,9%) dan 65 Legendary (8,1%). Model yang selalu menebak "Non-Legendary" sudah mendapat accuracy sekitar 92%, sehingga accuracy tidak cukup untuk menilai model. |
| Box plot fitur per kelas | Pada keenam statistik, median Legendary lebih tinggi daripada non-Legendary. Perbedaan paling jelas pada Sp. Atk dan Attack. Kelas non-Legendary punya banyak outlier ke atas, terutama pada HP dan Defense. |
| Heatmap korelasi | `Total` berkorelasi kuat dengan statistik penyusunnya (Sp. Atk 0,75, Attack 0,74, Sp. Def 0,72), sehingga berpotensi multikolinearitas. Terhadap `Legendary`, korelasi tertinggi ada pada `Total` (0,50) dan Sp. Atk (0,45), diikuti Sp. Def (0,36), Attack (0,35), dan Speed (0,33). `Generation` hampir tidak berkorelasi dengan fitur lain. |

**Implikasi untuk tahap berikutnya:** ketidakseimbangan kelas dan outlier ditangani di B3, sedangkan multikolinearitas `Total` dibahas di B4.

## B3. Preprocessing

### Keputusan preprocessing (B3)

| Langkah | Hasil pengecekan | Keputusan dan alasan |
|---|---|---|
| Missing values | Hanya `Type 2` yang kosong (386 dari 800 baris) | Diisi kategori "None". Kosong di sini berarti Pokémon hanya punya satu tipe, bukan data hilang, jadi baris tidak dibuang. |
| Data duplikat | 0 baris duplikat penuh | Tidak ada baris dihapus. Nomor `#` yang berulang berasal dari bentuk Mega dengan statistik berbeda, jadi dipertahankan sebagai data sah. |
| Outlier | Dideteksi dengan IQR: HP 19, Defense 13, Sp. Atk 10, Attack 7, Sp. Def 7, Speed 2 | Dipertahankan. Outlier adalah Pokémon nyata dengan statistik ekstrem (bukan kesalahan input), dan banyak di antaranya justru Legendary yang menjadi target. Jumlahnya juga kecil dibanding 800 baris. |
| Target | `Legendary` bertipe True/False | Diubah menjadi 1/0. |
| Encoding | `Type 1` dan `Type 2` kategorikal | One-hot encoding dengan `drop_first=True`. |
| Scaling | Fitur numerik berada pada rentang yang mirip, tetapi koefisien perlu dibandingkan antar fitur | `StandardScaler`, di-fit hanya pada data latih setelah pembagian data, untuk mencegah *data leakage*. |
| Kelas tidak seimbang | 735 : 65 (sekitar 92 : 8) | `class_weight='balanced'` pada Logistic Regression. Dipilih dibanding resampling karena data kecil (800 baris), sehingga tidak ingin membuang data atau membuat data sintetis. |

Tanpa penanganan, model cenderung mengabaikan kelas Legendary karena menebak semua baris sebagai non-Legendary sudah memberi accuracy tinggi (sekitar 92%).

## B4. Pemilihan Fitur

Fitur dipilih dengan tiga analisis: (1) multikolinearitas dengan VIF, (2) F-test (`SelectKBest`) untuk fitur numerik, (3) chi-square untuk fitur kategorikal.

| Fitur | VIF dengan `Total` | VIF tanpa `Total` | Skor F | Keputusan |
|---|---|---|---|---|
| `Total` | sekitar 1,0 x 10¹⁵ | - | - | Dibuang (jumlah dari enam statistik lain, redundan) |
| `Sp. Atk` | sekitar 1,0 x 10¹⁵ | 1,7 | 201,4 | Dipertahankan |
| `Sp. Def` | sekitar 1,0 x 10¹⁵ | 1,9 | 121,8 | Dipertahankan |
| `Attack` | sekitar 1,0 x 10¹⁵ | 1,8 | 108,1 | Dipertahankan |
| `Speed` | sekitar 1,0 x 10¹⁵ | 1,5 | 95,4 | Dipertahankan |
| `HP` | sekitar 1,0 x 10¹⁵ | 1,4 | 64,6 | Dipertahankan |
| `Defense` | sekitar 1,0 x 10¹⁵ | 1,7 | 51,6 | Dipertahankan |
| `Generation` | - | - | 5,1 (p = 0,024) | Dibuang (skor F jauh paling rendah, dan hanya penanda generasi rilis, bukan ukuran kekuatan) |
| `Type 1`, `Type 2` | - | - | chi-square tertinggi: Dragon 37,0; Psychic 20,6; Flying 9,4 | Dipertahankan (beberapa tipe signifikan, p < 0,05) |
| `#`, `Name` | - | - | - | Dibuang (hanya identitas, tidak membawa informasi kekuatan) |

`Total` memiliki VIF yang praktis tak hingga karena merupakan jumlah persis dari enam statistik lainnya (muncul peringatan *singular matrix* saat VIF dihitung, yang justru membuktikan redundansi ini). Setelah `Total` dibuang, VIF keenam statistik turun ke 1,4 sampai 1,9 (jauh di bawah batas 5), sehingga tidak ada multikolinearitas yang mengganggu. Semua statistik memiliki p-value mendekati 0 pada F-test. Fitur akhir: 6 statistik numerik ditambah hasil one-hot encoding `Type 1` dan `Type 2`, total **41 kolom**, dengan 800 baris.

## B5. Pembagian Data

Data dibagi dengan `train_test_split` rasio **80:20**, `stratify=y`, dan `random_state=42`.

* **Total sampel**: 800 baris
* **Data latih**: 640 sampel (80%), terdiri dari 588 non-Legendary dan 52 Legendary
* **Data uji**: 160 sampel (20%), terdiri dari 147 non-Legendary dan 13 Legendary
* **Fitur (X)**: HP, Attack, Defense, Sp. Atk, Sp. Def, Speed, serta hasil one-hot `Type 1` dan `Type 2` (41 kolom)
* **Target (y)**: `Legendary`

**Analisis:**
* **`stratify=y`**: proporsi Legendary tetap 8,125% di data latih (52 dari 640) maupun data uji (13 dari 160). Tanpa ini, data uji yang kecil bisa kebetulan berisi sangat sedikit Legendary sehingga evaluasi tidak adil.
* **`random_state=42`**: mengunci pengacakan agar hasil dapat direproduksi.
* **Pencegahan data leakage**: `StandardScaler` di-fit hanya pada data latih, lalu diterapkan ke data uji.

## B6. Pemodelan Logistic Regression

Model: `LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)`, dilatih pada 640 data latih.

* **Intercept**: -7,33

| Urutan | Fitur | Koefisien | Odds ratio | Arah dan tingkat pengaruh |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `Speed` | +1,923 | 6,85 | Positif, sangat kuat |
| 2 | `Sp. Def` | +1,722 | 5,60 | Positif, sangat kuat |
| 3 | `Sp. Atk` | +1,600 | 4,96 | Positif, sangat kuat |
| 4 | `HP` | +1,476 | 4,38 | Positif, kuat |
| 5 | `Type 1_Psychic` | +1,458 | 4,30 | Positif, kuat |
| 6 | `Defense` | +1,140 | 3,13 | Positif, kuat |
| ... | ... | ... | ... | ... |
| terendah 3 | `Type 2_Ground` | -0,981 | 0,375 | Negatif |
| terendah 2 | `Type 2_Dragon` | -1,192 | 0,304 | Negatif |
| terendah 1 | `Type 1_Normal` | -1,784 | 0,168 | Negatif, paling kuat |

(Enam fitur teratas dan tiga fitur terendah; 32 fitur lain berada di antara keduanya.)

**Cara membaca:** koefisien positif (odds ratio di atas 1) berarti kenaikan fitur meningkatkan peluang Pokémon menjadi Legendary, dan koefisien negatif sebaliknya. Pada fitur numerik yang sudah distandarisasi, odds ratio menunjukkan perubahan odds untuk kenaikan 1 standar deviasi, dengan fitur lain dianggap tetap. Pada kolom tipe (0/1), odds ratio membandingkan Pokémon bertipe tersebut dengan tipe acuan yang dibuang oleh `drop_first=True` (kategori pertama secara alfabet, yaitu Bug).

**Interpretasi:**
* **Speed** memiliki pengaruh terbesar: kenaikan 1 standar deviasi Speed meningkatkan odds menjadi Legendary sekitar 6,8 kali, diikuti **Sp. Def** (sekitar 5,6 kali) dan **Sp. Atk** (sekitar 5,0 kali). Statistik pertahanan khusus dan serangan khusus yang tinggi sangat mencirikan Legendary.
* **HP** (4,4 kali) dan **Defense** (3,1 kali) juga berpengaruh positif, walau lebih kecil.
* **Type 1_Psychic** meningkatkan odds Legendary sekitar 4,3 kali dibanding tipe acuan, konsisten dengan hasil chi-square di B4 (Psychic termasuk tipe teratas).
* Pengaruh paling negatif: **Type 1_Normal** (odds menjadi 0,17 kali), **Type 2_Dragon** (0,30 kali), dan **Type 2_Ground** (0,38 kali). Setelah statistik diperhitungkan, Pokémon bertipe ini lebih jarang Legendary dibanding tipe acuan.
* Urutan koefisien berbeda dari urutan skor F di B4 (Sp. Atk teratas pada F-test, tetapi Speed teratas pada koefisien). Ini wajar karena koefisien menunjukkan pengaruh sebuah fitur dengan fitur lain dianggap tetap, sedangkan F-test melihat tiap fitur secara terpisah.
* **Intercept** -7,33 adalah log-odds Legendary untuk Pokémon dengan statistik rata-rata dan tipe acuan, yang sangat rendah sesuai kenyataan bahwa Legendary hanya sekitar 8% data.

## B7. Evaluasi Model

Evaluasi dilakukan pada data uji (160 baris, 13 Legendary).

| Metrik | Nilai |
| :--- | :---: |
| Accuracy | 0,900 |
| Precision (Legendary) | 0,440 |
| Recall (Legendary) | 0,846 |
| F1-score (Legendary) | 0,579 |
| AUC | 0,947 |

**Confusion matrix:**

| | Ditebak Non-Legendary | Ditebak Legendary |
| :--- | :---: | :---: |
| **Aslinya Non-Legendary** | 133 (TN) | 14 (FP) |
| **Aslinya Legendary** | 2 (FN) | 11 (TP) |

**Kurva ROC:** berada jauh di atas garis tebakan acak dengan AUC 0,947, sehingga model mampu membedakan Pokémon Legendary dan non-Legendary dengan baik.

## B8. Analisis Performa

### 1. Metrik yang paling relevan
* **Accuracy kurang cocok** dijadikan acuan. Pada data uji, 147 dari 160 baris adalah non-Legendary, sehingga model yang selalu menebak "Non-Legendary" mendapat accuracy 91,9%, bahkan sedikit di atas model kita (90,0%), tetapi tidak menemukan satu pun Legendary.
* **Recall (0,846)** paling relevan karena tujuan utama adalah menemukan Legendary yang ada: 11 dari 13 berhasil ditemukan.
* **Precision (0,440)** perlu diperhatikan: dari 25 Pokémon yang ditebak Legendary, hanya 11 yang benar.
* **F1-score (0,579)** menyeimbangkan keduanya, dan **AUC (0,947)** menunjukkan kemampuan pemisahan kelas yang baik di berbagai ambang batas.

### 2. Jenis kesalahan (confusion matrix)
* Terdapat **14 false positive** (Pokémon biasa yang ditebak Legendary) dan **2 false negative** (Legendary yang terlewat). Kesalahan terbanyak adalah **false positive**.
* Ini konsekuensi `class_weight='balanced'`: model lebih berani menebak Legendary, sehingga recall naik tetapi precision turun.
* False positive kemungkinan terjadi pada Pokémon biasa berstatistik sangat tinggi (outlier di box plot, misalnya bentuk Mega atau evolusi akhir) yang polanya mirip Legendary.

### 3. Fitur yang paling berpengaruh
Berdasarkan koefisien dan odds ratio (B6), fitur paling berpengaruh terhadap peluang Legendary adalah `Speed` (odds ratio 6,85), `Sp. Def` (5,60), dan `Sp. Atk` (4,96), diikuti `HP` (4,38), tipe `Psychic` (4,30), dan `Defense` (3,13). Sebaliknya, tipe `Normal` (0,17), `Type 2_Dragon` (0,30), dan `Type 2_Ground` (0,38) menurunkan peluang Legendary. Pokémon Legendary dicirikan oleh kecepatan dan statistik khusus yang tinggi.

### 4. Keterbatasan model
1. **Data uji kecil**: hanya 13 Legendary, sehingga satu kesalahan mengubah recall sekitar 7,7 poin persentase dan hasil bisa berbeda jika split diacak ulang.
2. **Bentuk Mega**: variasi seperti Mega dicatat sebagai baris terpisah dengan statistik mirip bentuk aslinya. Jika satu bentuk masuk data latih dan bentuk lain masuk data uji, performa bisa tampak lebih baik dari sebenarnya.
3. **Status Legendary tidak sepenuhnya ditentukan statistik**: kategori ini ditentukan oleh alur cerita dan aturan game.
4. **Asumsi linearitas**: Logistic Regression tidak menangkap pola non-linear. Model seperti Random Forest bisa dicoba sebagai pembanding.
5. **Ambang klasifikasi 0,5 belum dioptimalkan**: menaikkannya dapat menaikkan precision dengan konsekuensi recall turun.
