# Laporan Proyek Machine Learning - Monica Mamondol
## Domain Proyek
Domain yang dipilih untuk proyek *machine learning* ini adalah **Kesehatan**, dengan judul **Predictive Analytics: Klasifikasi Tumor Payudara**  
### Latar Belakang

![foto wanita](https://i.ibb.co.com/PZvPJwwk/Screenshot-2025-04-25-184045.png)

Tumor payudara adalah kondisi di mana sel-sel di jaringan payudara tumbuh secara berlebihan dan tidak terkendali sehingga membentuk massa atau benjolan yang tidak normal. Tumor ini terbagi menjadi dua jenis, yaitu tumor jinak dan tumor ganas (kanker). Tumor jinak biasanya tumbuh secara terbatas, tidak menyebar ke jaringan lain, dan dapat diangkat secara utuh, sedangkan tumor ganas memiliki kemampuan untuk menginfiltrasi jaringan sekitar, menyebar (metastasis), dan berkembang lebih cepat. [[1](https://www.newneraca.neraca.co.id/article/217453/membangun-edukasi-masyarakat-deteksi-dini-jadi-kunci-kesembuhan-kanker-payudara)] Banyak pasien datang pada stadium lanjut karena kurangnya kesadaran atau ketakutan sehingga deteksi dini menjadi sulit. Pemeriksaan fisik, mammografi, USG, dan biopsi diperlukan untuk diagnosis yang akurat, namun keterbatasan akses dan ketidaktahuan pasien menghambat pemeriksaan awal. [[2](http://repo.unand.ac.id/13194/1/Laporan%20Kemajuan%20EHealth%20Deteksi%20kanker%20Payudara.pdf)] Oleh karena itu, machine learning dapat membantu mengatasi tantangan ini dengan meningkatkan akurasi dan kecepatan diagnosis melalui analisis citra medis seperti mammografi dan USG. Algoritma machine learning dapat mendeteksi pola-pola halus pada gambar payudara yang sulit dikenali oleh mata manusia. Dengan kemampuan klasifikasi yang baik, predictive analytics membantu membedakan antara tumor jinak dan ganas secara cepat dan tepat, sehingga mempercepat proses diagnosis dan pengambilan keputusan klinis.

Mengapa Masalah Ini Harus Diselesaikan?
- Peningkatan efisiensi diagnosis : Model machine learning dapat memberikan prediksi awal secara instan, sehingga membantu tenaga medis mengambil keputusan lebih cepat.
- Mengurangi risiko error manusia : Diagnosis manual rentan terhadap variasi interpretasi antar-ahli.
- Akses layanan kesehatan yang lebih luas : Sistem otomatis dapat diterapkan di fasilitas kesehatan dengan sumber daya terbatas.
- Deteksi dini yang lebih baik : Prediksi yang cepat dan tepat berkontribusi besar dalam meningkatkan tingkat kesembuhan pasien.

## Business Understanding
Pengembangan model prediksi untuk klasifikasi tumor payudara memiliki potensi besar untuk memberikan manfaat bagi berbagai pihak, termasuk pasien, tenaga medis, dan sistem kesehatan secara keseluruhan. Model ini dapat membantu meningkatkan akurasi diagnosis,  serta mendukung efisiensi dalam proses deteksi dini penyakit.
### Problem Statements
Berdasarkan latar belakang di atas, berikut ini merupakan rincian masalah yang dapat diselesaikan pada proyek ini:
- Algoritma machine learning manakah yang paling efektif dalam menghasilkan prediksi yang akurat dan konsisten untuk kasus tumor payudara?
- Apakah jenis tumor dapat diklasifikasikan secara akurat berdasarkan fitur-fitur numerik hasil biopsi jaringan?
- Bagaimana tingkat akurasi dan reliabilitas prediksi model machine learning untuk membedakan antara tumor jinak dan ganas?
### Goals
Tujuan dari proyek ini adalah:
- Membandingkan efektivitas beberapa algoritma machine learning (Random Forest, Logistic Regression, dan SVM) untuk menemukan model terbaik dalam prediksi jenis tumor.
- Membuktikan apakah jenis tumor (jinak/ganas) dapat diklasifikasikan secara akurat menggunakan fitur numerik dari dataset WDBC melalui pendekatan machine learning.
- Menilai tingkat akurasi, reliabilitas, dan performa model machine learning dalam membedakan tumor jinak dan ganas menggunakan metrik evaluasi seperti accuracy, precision, recall, F1-score, dan confusion matrix.

### Solution Statements
 Membuat beberapa alternatif solusi untuk mendapatkan model yang paling baik dari beberapa model yang telah dibuat untuk klasifikasi tumor payudara. Diantaranya adalah menggunakan:
* Random Forest adalah algoritma machine learning yang kuat yang dapat digunakan untuk berbagai tugas klasifikasi. Ini adalah metode ensemble, yang berarti bahwa model random forest terdiri dari banyak decision tree kecil, yang disebut estimator, yang masing-masing menghasilkan prediksi mereka sendiri. [[3](https://deepai.org/machine-learning-glossary-and-terms/random-forest)]  
* Metrik Evaluasi :
Akurasi.
Recall (untuk meminimalkan false negative).
F1-score (seimbang antara precision dan recall).
- Logistic Regression adalah salah satu metode supervised learning dalam machine learning yang digunakan untuk memprediksi probabilitas suatu kejadian kategori atau kelas. [[4](https://aws.amazon.com/id/what-is/logistic-regression/)]
- Metrik Evaluasi :
Precision (menghindari overdiagnosis).
Accuracy.
Confusion Matrix untuk melihat jumlah TP, TN, FP, FN.
* Support Vector Machine (SVM) adalah algoritma yang digunakan untuk menemukan hyperplane dalam ruang N-dimensi (N - jumlah fitur) yang secara jelas mengklasifikasikan titik data. SVM sangat handal dalam menyelesaikan masalah klasifikasi dengan boundary yang kompleks, seperti pada dataset tumor payudara. [[5](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)]
* Metrik Evaluasi :
Specificity (kemampuan mendeteksi true negative).
Akurasi keseluruhan.

## Data Understanding
### EDA - Deskripsi Variabel
**Informasi Datasets**

| Jenis | Keterangan |
| ------ | ------ |
| Title | _Breast Cancer Wisconsin (Diagnostic) Data Set_ |
| Source | [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data) |
| Maintainer | [UCI Machine Learning](https://www.kaggle.com/organizations/uciml) |
| License | Other (specified in description) |
| Visibility | Publik |
| Tags | _Cancer, Healthcare_ |
| Usability | 8.53 |

Berikut informasi pada dataset: 
Nama Datasets: _Breast Cancer Wisconsin (Diagnostic) Data Set_

| id      | diagnosis | radius_mean | texture_mean | perimeter_mean | area_mean | smoothness_mean | compactness_mean | concavity_mean | concave points_mean | symmetry_mean | fractal_dimension_mean | radius_se | texture_se | perimeter_se | area_se | smoothness_se | compactness_se | concavity_se | concave points_se | symmetry_se | fractal_dimension_se | radius_worst | texture_worst | perimeter_worst | area_worst | smoothness_worst | compactness_worst | concavity_worst | concave points_worst | symmetry_worst | fractal_dimension_worst | Unnamed: 32 |
|---------|-----------|-------------|--------------|----------------|-----------|-----------------|------------------|----------------|---------------------|---------------|------------------------|-----------|------------|--------------|---------|---------------|----------------|--------------|-------------------|-------------|----------------------|--------------|---------------|-----------------|------------|------------------|-------------------|-----------------|----------------------|----------------|--------------------------|-------------|
| 842302  | M         | 17.99       | 10.38        | 122.8          | 1001      | 0.1184          | 0.2776           | 0.3001         | 0.1471              | 0.2419        | 0.07871                | 1.095     | 0.9053     | 8.589        | 153.4   | 0.006399      | 0.04904        | 0.05373      | 0.01587           | 0.03003     | 0.006193             | 25.38       | 17.33         | 184.6           | 2019       | 0.1622           | 0.6656            | 0.7119          | 0.2654               | 0.4601         | 0.1189                   | NaN         |
| 842517  | M         | 20.57       | 17.77        | 132.9          | 1326      | 0.08474         | 0.07864          | 0.0869         | 0.07017             | 0.1812        | 0.05667                | 0.5435    | 0.7339     | 3.398        | 74.08   | 0.005225      | 0.01308        | 0.0186       | 0.0134            | 0.01389     | 0.003532             | 24.99       | 23.41         | 158.8           | 1956       | 0.1238           | 0.1866            | 0.2416          | 0.186                | 0.275          | 0.08902                  | NaN         |
| 84300903| M         | 19.69       | 21.25        | 130            | 1203      | 0.1096          | 0.1599           | 0.1974         | 0.1279              | 0.2069        | 0.05999                | 0.7456    | 0.7869     | 4.585        | 94.03   | 0.00615       | 0.04006        | 0.03832      | 0.02058           | 0.0225      | 0.004571             | 23.57       | 25.53         | 152.5           | 1709       | 0.1444           | 0.4245            | 0.4504          | 0.243                | 0.3613         | 0.08758                  | NaN         |
| 84348301| M         | 11.42       | 20.38        | 77.58          | 386.1     | 0.1425          | 0.2839           | 0.2414         | 0.1052              | 0.2597        | 0.09744                | 0.4956    | 1.156      | 3.445        | 27.23   | 0.00911       | 0.07458        | 0.05661      | 0.01867           | 0.05963     | 0.009208             | 14.91       | 26.5          | 98.87           | 567.7      | 0.2098           | 0.8663            | 0.6869          | 0.2575               | 0.6638         | 0.173                    | NaN         |
| 84358402| M         | 20.29       | 14.34        | 135.1          | 1297      | 0.1003          | 0.1328           | 0.198          | 0.1043              | 0.1809        | 0.05883                | 0.7572    | 0.7813     | 5.438        | 94.44   | 0.01149       | 0.02461        | 0.05688      | 0.01885           | 0.01756     | 0.005115             | 22.54       | 16.67         | 152.2           | 1575       | 0.1374           | 0.205             | 0.4             | 0.1625               | 0.2364         | 0.07678                  | NaN         |
| 843786  | M         | 12.45       | 15.7         | 82.57          | 477.1     | 0.1278          | 0.17             | 0.1578         | 0.08089             | 0.2087        | 0.07613                | 0.3345    | 0.8902     | 2.217        | 27.19   | 0.00751       | 0.03345        | 0.03672      | 0.01137           | 0.02165     | 0.005082             | 15.47       | 23.75         | 103.4           | 741.6      | 0.1791           | 0.5249            | 0.5355          | 0.1741               | 0.3985         | 0.1244                   | NaN         |

Tabel 1. EDA Deskripsi Variabel

Dilihat dari _Tabel 1. EDA Deskripsi Variabel_ dataset ini telah di *bersihkan* terlebih dahulu oleh pembuat, sehingga mudah digunakan dan ramah bagi pemula. 
- Dataset berupa CSV (Comma-Seperated Values).
- Dataset memiliki 569 sample dengan 33 fitur.
- Dataset memiliki 32 fitur bertipe float64 dan 1 fitur bertipe object.
- Dataset ini tidak terdapat data yang terduplikat.
- Dataset ini tidak ada missing value.

### Variable - variable pada dataset
- `id` : Nomor identitas unik untuk setiap sampel (pasien).
- `diagnosis` : Label kelas untuk tumor, menunjukkan apakah tumor bersifat jinak (B) atau ganas (M).
- `radius_mean` : Rata-rata jarak dari pusat ke titik-titik pada perimeter sel.
- `texture_mean` : Standar deviasi intensitas piksel dalam gambar sel.
- `perimeter_mean` : Rata-rata keliling sel.
- `area_mean` : Rata-rata luas area sel.
- `smoothness_mean` : Variasi lokal dalam panjang radius sel.
- `compactness_mean` : Rasio antara perimeter kuadrat dan area sel.
- `concavity_mean` : Seberapa cekung kontur sel.
- `concave points_mean` : Jumlah titik cekung pada kontur sel.
- `symmetry_mean` : Tingkat simetri sel.
- `fractal_dimension_mean` : pendekatan kurva terhadap dimensi fraktal.
-  `radius_se` : Variabilitas pengukuran radius rata-rata.
-  `texture_se` : Variasi dalam pengukuran tekstur.
-   `perimeter_se` : Ketidakpastian panjang perimeter.
-   `area_se` : Variasi luas area nukleus.
-   `smoothness_se` : Konsistensi kelicinan permukaan.
-   `compactness_se` : Variasi kekompakan bentuk.
-   `concavity_se` : Ketidakkonsistenan bagian cekung.
-   `concave points_se` : Variasi jumlah titik cekung.
-   `symmetry_se` : Fluktuasi tingkat simetri.
-   `fractal_dimension_se` : Variasi kompleksitas fraktal.
-   `radius_worst` : Radius terbesar yang terukur.
-   `texture_worst` : Tekstur paling tidak homogen.
-   `perimeter_worst` : Perimeter terpanjang.
-   `area_worst` : Luas terbesar.
-   `smoothness_worst` : Kelicinan paling tidak konsisten.
-   `compactness_worst` : Bentuk paling tidak kompak.
-   `concavity_worst` : Cekungan terdalam.
-   `concave points_worst` : Jumlah titik cekung terbanyak.
-   `symmetry_worst` : Simetri paling buruk.
-   `fractal_dimension_worst` : Kompleksitas fraktal tertinggi.
-   `Unamed:32` : seluruhnya berisi nilai NaN (tidak ada data).


### EDA - Univariate Analysis

![Univariate Analysis](https://i.ibb.co.com/Y7yk84TZ/Screenshot-2025-04-28-165754.png)

Gambar 1. Analisis Univariat (Data Numerik)

pada gambar 1.  data numerik memiliki karakteristik, yaitu:
- Beberapa fitur seperti radius_mean, perimeter_mean, dan area_mean memiliki distribusi yang hampir normal. Ini berarti nilai-nilai pada fitur tersebut tersebar secara simetris di sekitar mean.
- Beberapa fitur seperti concavity_mean, concave points_mean, dan fractal_dimension_mean memiliki distribusi yang sangat skew (tidak simetris). Nilai-nilai cenderung terkonsentrasi di satu sisi, dengan ekor panjang ke arah nilai tertentu.
- Beberapa fitur seperti radius_worst, perimeter_worst, dan area_worst memiliki rentang nilai yang sangat lebar. Histogram menunjukkan bahwa ada beberapa outlier atau nilai ekstrem yang jauh lebih besar daripada nilai-nilai lainnya.
- Beberapa fitur seperti symmetry_mean, fractal_dimension_mean, symmetry_se, dan fractal_dimension_se memiliki distribusi yang relatif kompak, dengan nilai-nilai yang cenderung berkumpul di sekitar nilai tertentu.
- Beberapa fitur seperti texture_se, perimeter_se, dan area_se memiliki distribusi yang unik, dengan beberapa puncak (modus) yang jelas. Ini bisa menjadi indikasi adanya kelompok-kelompok data yang berbeda dalam dataset. 

### EDA - Multivariate Analysis

![Multivariate Analysis](https://i.ibb.co.com/zTs84xP5/Screenshot-2025-04-29-195109.png)

Gambar 2. Analisis Matriks Korelasi

Pada _Gambar 2. Analisis Matriks Korelasi_, merupakan _Correlation Matrix_ menunjukkan hubungan antar fitur dalam nilai korelasi. Jika diamati, terlihat bahwa beberapa fitur memiliki korelasi yang sangat tinggi satu sama lain. Misalnya, `radius_mean`, `perimeter_mean`, dan `area_mean` memiliki korelasi >0.9.

## Data Preparation
Tahap data preparation dilakukan agar data siap digunakan untuk proses modeling machine learning. Tanpa persiapan yang baik, model cenderung tidak akurat atau tidak stabil karena adanya masalah seperti missing values, ketidakseimbangan data, atau skala data yang berbeda-beda.
- Pada dataset ini tidak ada missing value, dan duplikat 
- Kolom `id` dan `Unnamed: 32` dihapus karena tidak memiliki kontribusi dalam prediksi.
- Label kelas pada kolom diagnosis yang awalnya bertipe karakter (B, M) diubah menjadi numerik (0, 1).
- Pada _Train-Test-Split_ rasio pembagian data adalah 80% untuk data latih dan 20% untuk data uji.
- Pada proyek kasus ini digunakan _Normalization_ untuk menormalisasi dataset. Semua proses ini diperlukan dalam rangka membuat model yang baik.

| No  | Langkah                   | Tujuan / Alasan                                      |
|-----|---------------------------|-----------------------------------------------------|
| 1   | Menghapus kolom tidak relevan | Meningkatkan efisiensi dan fokus pada fitur penting |
| 2   | Encoding label            | Memenuhi syarat input numerik untuk model ML        |
| 3   | Membagi fitur dan target  | Persiapan untuk modeling dan evaluasi               |
| 4   | Train-Test Split         | Evaluasi objektif dan validasi model                |
| 5   | Feature Scaling (Standardisasi/Normalisasi)         | Meningkatkan kualitas dan stabilitas model ML       |

## Modeling

## Tahapan Pemodelan
1. **Mempersiapkan Data**  
   - Data telah melalui preprocessing (handling missing values, encoding, scaling, dll)
   - Dibagi menjadi training set (80%) dan test set (20%)

2. **Pemilihan Algoritma**  
   - Dipilih 3 algoritma yang cocok untuk klasifikasi:
     * Random Forest (ensemble method)
     * Logistic Regression (linear model)
     * SVM (kernel-based method)

3. **Pelatihan Model**  
   - Setiap model dilatih dengan data training
   - Menggunakan default parameter untuk baseline performance

4. **Evaluasi Model**  
   - Diukur menggunakan accuracy, precision, recall, dan f1-score
   - Confusion matrix untuk analisis lebih detail

## Analisis Model

### 1. Random Forest
**Cara Kerja:**  
- Bootstrapping: Setiap pohon dilatih pada subset data acak (sampling with replacement).
- Split Fitur Acak: Setiap split node menggunakan subset fitur acak (default: sqrt(n_features)).
- Agregasi Hasil: Prediksi akhir ditentukan oleh mayoritas voting dari semua pohon.

**Parameter:**  
- n_estimators=100 (default): Jumlah pohon dalam forest. Nilai default dipilih karena stabil dan cukup untuk akurasi tinggi.
- random_state=42: Untuk memastikan hasil dapat direproduksi.

**Kelebihan:**
- Tahan terhadap overfitting
- Handles non-linear relationships well
- Tidak membutuhkan feature scaling

**Kekurangan:**
- Lebih lambat dalam prediksi
- Kurang interpretabel dibanding model linear

### 2. Logistic Regression
**Cara Kerja:**  
- Transformasi Linear: z = b0 + b1*x1 + ... + bn*xn
- Fungsi Sigmoid: P(y=1) = 1 / (1 + e^(-z))
- Klasifikasi: Threshold default 0.5 menentukan kelas prediksi.

**Parameter:**  
- penalty='l2' (regularisasi Ridge).
- C=1.0 (kebalikan dari kekuatan regularisasi).

**Kelebihan:**
- Cepat dalam training dan prediksi
- Hasil mudah diinterpretasikan
- Probabilistic output

**Kekurangan:**
- Asumsi linearitas
- Sensitif terhadap outliers

### 3. Support Vector Machine (SVM)
**Cara Kerja:**  
- SVM mencari hyperplane optimal yang memisahkan kelas dengan margin terlebar. Untuk data non-linear, digunakan kernel trick.

**Parameter:**  
- kernel='rbf' (default): Kernel Radial Basis Function untuk menangani non-linearitas.
- probability=True: Memungkinkan estimasi probabilitas (untuk metode seperti predict_proba).


**Kelebihan:**
- Efektif di high dimensional space
- Robust terhadap overfitting
- Hasil terbaik dalam eksperimen ini

**Kekurangan:**
- Waktu training lebih lama
- Sulit diinterpretasikan
- Sensitif terhadap parameter tuning

## Pemilihan Model Terbaik

**Model Terpilih:** Support Vector Machine (SVM)

**Alasan Pemilihan:**
1. **Performance Terbaik**  
   - Accuracy tertinggi (98.25%)  
   - Precision sempurna (1.00) untuk kelas positif  
   - Zero false positives  

2. **Keseimbangan Metric**  
   - Memiliki recall dan f1-score yang seimbang  
   - Confusion matrix menunjukkan hanya 2 false negatives  

3. **Kemampuan Generalisasi**  
   - Walaupun tanpa tuning parameter, menunjukkan hasil yang konsisten  
   - Tidak ada misklasifikasi untuk kelas negatif (0 false positives)  

## Evaluation



| Problem Statement | Jawaban dari Model |
|-------------------|--------------------|
| **Algoritma mana yang paling efektif?** | SVM memberikan akurasi tertinggi (**98.25%**), diikuti Logistic Regression (**97.37%**) dan Random Forest (**96.49%**). |
| **Apakah fitur numerik bisa mengklasifikasi tumor?** | **Ya**, terbukti dengan akurasi >96% pada semua model. Fitur seperti `concavity_worst` dan `radius_mean` sangat berpengaruh (lihat `feature_importance` Random Forest). |
| **Bagaimana akurasi dan reliabilitas model?** | Semua model memiliki **precision >96%** dan **recall >93%** untuk kelas ganas (1). SVM mencapai **100% precision** untuk kelas ganas (tidak ada False Positive). |


| Goal                          | Pencapaian                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| Membandingkan algoritma       | SVM > Logistic Regression > Random Forest dalam hal akurasi.               |
| Membuktikan klasifikasi akurat| Akurasi model sangat tinggi (>96%) dengan konsistensi yang baik (lihat confusion matrix). |
| Menilai performa metrik       | - **SVM**: Recall = 95% (minim False Negative)  
|                               | - **Logistic Regression**: Precision = 98% (minim False Positive)  
|                               | - **Random Forest**: F1-score = 95% (seimbang)                             |

| Solusi              | Dampak                                                                                       |
|---------------------|-----------------------------------------------------------------------------------------------|
| **Random Forest**   | - Stabil dan interpretable (bisa analisis `feature_importance`).<br>- Cocok untuk data kompleks, tapi akurasi sedikit di bawah SVM. |
| **Logistic Regression** | - Sederhana dan cepat.<br>- Precision tinggi (98%), cocok untuk menghindari overdiagnosis (False Positive). |
| **SVM**             | - Solusi terbaik dengan akurasi tertinggi (98.25%).<br>- 100% precision untuk kelas ganas = tidak ada pasien sehat yang salah didiagnosis ganas. |

### Metrik Evaluasi yang Digunakan

### 1. Accuracy
**Formula:**  
`Accuracy = (TP + TN) / (TP + TN + FP + FN)`

**Penjelasan:**  
- Mengukur persentase prediksi yang benar dari seluruh prediksi. 
- Menggunakan semua elemen confusion matrix.
- Range nilai: 0 (terburuk) sampai 1 (terbaik).

### 2. Precision
**Formula:**  
`Precision = TP / (TP + FP)`

**Penjelasan:**  
- Mengukur proporsi prediksi positif yang benar.
- Fokus pada false positive (prediksi salah sebagai positif).
- Disebut juga positive predictive value.

### 3. Recall (Sensitivity)
**Formula:**  
`Recall = TP / (TP + FN)`

**Penjelasan:**  
- Mengukur kemampuan model mendeteksi instance positif. Krusial ketika False Negative berbahaya.
- Fokus pada false negative (positif yang terlewat).
- Disebut juga sensitivity atau hit rate.

### 4. F1-Score
**Formula:**  
`F1 = 2 * (Precision * Recall) / (Precision + Recall)`

**Penjelasan:**  
- Rata-rata harmonik precision dan recall. 
- Baik untuk dataset tidak seimbang.
- Lebih baik daripada accuracy untuk data imbalance

### 5. Confusion Matrix
**Komponen:**
- Tabel yang membandingkan prediksi vs aktual
- True Positive (TP)
- True Negative (TN)
- False Positive (FP)
- False Negative (FN)

**Penjelasan:**  
Memberikan breakdown lengkap tipe prediksi benar dan salah.

## Hasil Evaluasi Model

### Random Forest
| Metric  | Nilai  | Interpretasi |
|---------|--------|--------------|
| Accuracy| 96.49% | 110/114 prediksi benar |
| Precision | 0.98 | Hanya 1 FP |
| Recall | 0.93 | 3 FN terdeteksi |
| F1-Score | 0.95 | Keseimbangan baik |

### Logistic Regression
| Metric  | Nilai  | Interpretasi |
|---------|--------|--------------|
| Accuracy| 97.37% | 111/114 prediksi benar |
| Precision | 0.98 | 1 FP |
| Recall | 0.95 | 2 FN |
| F1-Score | 0.96 | Lebih baik dari RF |

### SVM
| Metric  | Nilai  | Interpretasi |
|---------|--------|--------------|
| Accuracy| 98.25% | 112/114 prediksi benar |
| Precision | 1.00 | 0 FP (sempurna) |
| Recall | 0.95 | 2 FN |
| F1-Score | 0.98 | Performa terbaik |

Berikut hasil accuracy 3 buah model yang latih:

| Model | Accuracy |
| ------ | ------ |
| RandomForest  | 0.96 |
| SVM | 0.98 |
| Logistic Regression | 0.97 |

![Plot Accuracy](https://i.ibb.co.com/WN83XVBH/Screenshot-2025-04-30-200441.png) 

Dilihat dari tabel dan gambar tersebut dapat diketahui bahwa model dengan algoritma SVM memiliki Accuracy yang lebih tinggi dengan accuracy 98% . Untuk itu model tersebut yang akan dipilih untuk digunakan.



## Referensi
1. https://www.newneraca.neraca.co.id/article/217453/membangun-edukasi-masyarakat-deteksi-dini-jadi-kunci-kesembuhan-kanker-payudara
2. http://repo.unand.ac.id/13194/1/Laporan%20Kemajuan%20EHealth%20Deteksi%20kanker%20Payudara.pdf
3. https://deepai.org/machine-learning-glossary-and-terms/random-forest)
4. https://aws.amazon.com/id/what-is/logistic-regression/
5. https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47

