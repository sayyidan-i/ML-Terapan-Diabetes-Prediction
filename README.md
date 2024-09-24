# Laporan Proyek Machine Learning - Sayyidan Muhamad Ikhsan
# Prediksi Diabetes

## Domain Proyek

Diabetes mellitus adalah penyakit kronis yang ditandai oleh kadar gula darah yang tinggi karena tubuh tidak dapat menghasilkan atau menggunakan insulin secara efektif. Insulin berperan dalam memindahkan glukosa dari darah ke dalam sel untuk digunakan sebagai energi. Di Indonesia, diabetes merupakan salah satu penyakit kronis paling umum, dengan prevalensi yang meningkat dari tahun ke tahun. Data Riskesdas 2018 menunjukkan peningkatan prevalensi diabetes dari 6,9% pada tahun 2013 menjadi 8,5% pada tahun 2018, menegaskan bahwa diabetes adalah masalah kesehatan serius yang memerlukan perhatian lebih lanjut.

Deteksi dini diabetes sangat penting karena alasan-alasan berikut. Pertama, deteksi dini memungkinkan intervensi segera untuk mencegah komplikasi serius seperti penyakit jantung, stroke, gagal ginjal, kebutaan, dan amputasi kaki. Kedua, pengelolaan yang efektif dapat meningkatkan kualitas hidup penderita dengan mengontrol kadar gula darah dan mencegah komplikasi. Ketiga, deteksi dini dapat mengurangi beban ekonomi dengan mencegah atau menunda komplikasi yang memerlukan perawatan medis yang mahal. Keempat, meningkatkan kesadaran tentang diabetes dan pentingnya gaya hidup sehat dapat membantu mencegah kasus baru di masa depan. Kelima, deteksi dini juga dapat membantu individu lain yang berisiko tinggi, seperti anggota keluarga, untuk mengambil langkah-langkah pencegahan yang tepat. Dengan demikian, pendeteksian dini diabetes bukan hanya memberi manfaat bagi individu yang terkena penyakit, tetapi juga bagi masyarakat secara keseluruhan dalam memerangi penyebaran diabetes.

## Business Understanding

### Problem Statements

- Bagaimana melakukan deteksi diabetes dini kepada seseorang berdasarkan fitur-fitur klinis tertentu?

### Goals

- Melakukan deteksi diabetes dini seseorang berdasarkan fitur-fitur klinis tertentu?

### Solution statements
- Menggunakan 3 model berbeda dan mengukur model yang memiliki performa terbaik dalam melakukan deteksi diabetes. Model yang digunakan adalah Decision Tree, Random Forest, dan XGBoost
- Ketiga model dievaluasi menggunakan dua metrik berbeda, yaitu:
  - Akurasi: persentase jumlah prediksi yang benar
  - Mean Squared Error (MSE): mengukur rata-rata dari kuadrat perbedaan antara nilai prediksi yang dihasilkan oleh model dengan nilai sebenarnya dari data yang diamati.  MSE memberikan gambaran tentang seberapa dekat prediksi model dengan nilai sebenarnya, dengan nilai MSE yang lebih rendah menunjukkan bahwa model memiliki tingkat kesalahan yang lebih rendah dan memberikan prediksi yang lebih baik.

## Data Understanding
Data yang digunakan berasal dari **UCI Machine Learning Repository** dengan judul [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)


### Variabel-variabel pada CDC Diabetes Health Indicator dataset adalah sebagai berikut:

| Variable Name       | Role              | Type    | Description                                                                                      |
|---------------------|-------------------|---------|--------------------------------------------------------------------------------------------------|                                                                                       
| Diabetes_binary     | Target            | Binary  | 0 = no diabetes 1 = prediabetes or diabetes                                                    |
| HighBP              | Feature           | Binary  | 0 = no high BP 1 = high BP                                                                      |
| HighChol            | Feature           | Binary  | 0 = no high cholesterol 1 = high cholesterol                                                    |
| CholCheck           | Feature           | Binary  | 0 = no cholesterol check in 5 years 1 = yes cholesterol check in 5 years                        |
| BMI                 | Feature           | Integer | Body Mass Index                                                                                  |
| Smoker              | Feature           | Binary  | Have you smoked at least 100 cigarettes in your entire life? 0 = no 1 = yes                      |
| Stroke              | Feature           | Binary  | (Ever told) you had a stroke. 0 = no 1 = yes                                                    |
| HeartDiseaseorAttack | Feature         | Binary  | Coronary heart disease (CHD) or myocardial infarction (MI) 0 = no 1 = yes                        |
| PhysActivity        | Feature           | Binary  | Physical activity in past 30 days - not including job 0 = no 1 = yes                             |
| Fruits              | Feature           | Binary  | Consume Fruit 1 or more times per day 0 = no 1 = yes                                            |
| Veggies             | Feature           | Binary  | Consume Vegetables 1 or more times per day 0 = no 1 = yes                                       |
| HvyAlcoholConsump   | Feature           | Binary  | Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week) 0 = no 1 = yes |
| AnyHealthcare       | Feature           | Binary  | Have any kind of health care coverage, including health insurance, prepaid plans such as HMO, etc. 0 = no 1 = yes |
| NoDocbcCost         | Feature           | Binary  | Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? 0 = no 1 = yes |
| GenHlth             | Feature           | Integer | Would you say that in general your health is: scale 1-5 1 = excellent 2 = very good 3 = good 4 = fair 5 = poor |
| MentHlth            | Feature           | Integer | Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? scale 1-30 days |
| PhysHlth            | Feature           | Integer | Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? scale 1-30 days |
| DiffWalk            | Feature           | Binary  | Do you have serious difficulty walking or climbing stairs? 0 = no 1 = yes                         |
| Sex                 | Feature           | Binary  | Sex 0 = female 1 = male                                                                          |
| Age                 | Feature           | Integer | Age 13-level age category (_AGEG5YR see codebook) 1 = 18-24 9 = 60-64 13 = 80 or older            |
| Education           | Feature           | Integer | Education level (EDUCA see codebook) scale 1-6 1 = Never attended school or only kindergarten 2 = Grades 1 through 8 (Elementary) 3 = Grades 9 through 11 (Some high school) 4 = Grade 12 or GED (High school graduate) 5 = College 1 year to 3 years (Some college or technical school) 6 = College 4 years or more (College graduate) |
| Income              | Feature           | Integer | Income scale (INCOME2 see codebook) scale 1-8 1 = less than $10,000 5 = less than $35,000 8 = $75,000 or more |

## Data Preparation
Data Preparation yang dilakukan adalah
1. Melihat apakah terdapat missing value dan perlu diatasi. 
   
   Missing value perlu diatasi karena dapat mempengaruhi kualitas dan hasil dari analisis data. Jika missing value tidak diatasi, dapat menyebabkan bias dalam model yang dibangun dan mengurangi akurasi prediksi. Selain itu, beberapa algoritma machine learning tidak dapat menghandle missing value secara langsung, sehingga perlu dilakukan pengisian atau penghapusan missing value sebelum data dapat digunakan untuk melatih model. Terdapat beberapa metode yang dapat digunakan untuk mengatasi missing value, seperti pengisian dengan nilai rata-rata, median, modus, atau menggunakan teknik imputasi seperti regresi atau algoritma machine learning lainnya.
2. Memilih 15 fitur yang memiliki korelasi tertinggi terhadap prediksi yang akan dilatih
    
    Hal ini membantu meningkatkan kinerja model dengan menggunakan informasi yang paling relevan untuk memprediksi target. Fokus pada fitur-fitur yang memiliki korelasi tinggi juga dapat mempercepat waktu pelatihan model dengan mengurangi dimensi data yang tidak relevan. Selain itu, pemilihan fitur-fitur yang tepat juga membantu mengurangi risiko overfitting, di mana model terlalu sesuai dengan data pelatihan dan kehilangan kemampuan untuk menggeneralisasi pada data baru. Dengan demikian, menentukan fitur-fitur yang optimal untuk dilatih adalah langkah penting dalam pengembangan model yang efektif.

3. Melakukan normalisasi data

    Normalisasi data penting dalam pemrosesan data karena memastikan bahwa semua fitur memiliki skala yang serupa, sehingga tidak ada fitur yang mendominasi yang lainnya. Ini membantu algoritma pembelajaran mesin konvergen lebih cepat dan mencegah bobot yang tidak seimbang antara fitur-fitur. Normalisasi juga membantu meningkatkan performa model, terutama untuk algoritma yang sensitif terhadap skala, seperti regresi linier dan jaringan saraf tiruan. Salah satu metode normalisasi yang umum digunakan adalah Min-Max Scaling.

    Min-Max Scaling, atau MinMaxScaler dalam library Scikit-learn (sklearn), adalah metode normalisasi yang mentransformasikan fitur-fitur sehingga rentang nilainya berada dalam interval yang ditentukan, biasanya antara 0 dan 1. Proses ini dilakukan dengan mengurangi nilai minimum dari setiap fitur dan membaginya dengan selisih antara nilai maksimum dan minimum. Metode ini menjaga hubungan relatif antara nilai-nilai fitur sambil memperbaiki distribusi datanya.

4. Melakukan split dataset 80:20 untuk data latih dan data uji
    
    Pentingnya membagi data menjadi data training dan data testing adalah untuk menguji kinerja model machine learning secara objektif dan menghindari overfitting. Data training digunakan untuk melatih model, sedangkan data testing digunakan untuk mengevaluasi seberapa baik model tersebut dapat melakukan prediksi pada data yang tidak digunakan selama pelatihan.

## Modeling
Model machine learning yang digunakan adalah Decision Tree, Random Forest, dan XGBoost.
1. Decision Tree

    Decision tree adalah salah satu metode dalam machine learning yang digunakan untuk melakukan pemodelan prediktif dan mengambil keputusan berdasarkan serangkaian aturan yang dihasilkan dari data latih. Pada dasarnya, decision tree menggambarkan struktur pohon di mana setiap simpul internal mewakili keputusan berdasarkan fitur-fitur data, sedangkan cabang-cabangnya merepresentasikan hasil dari keputusan tersebut. Proses pembuatan decision tree melibatkan pemilihan fitur yang paling informatif untuk membagi data secara rekursif sehingga setiap cabang dari pohon meminimalkan ketidakmurnian (misclassification). Keuntungan utama dari decision tree adalah kemampuannya untuk menghasilkan aturan yang mudah diinterpretasi, karena representasi visualnya yang mirip dengan alur pemikiran manusia. Namun, decision tree cenderung rentan terhadap overfitting jika tidak diatur dengan baik, dan untuk meningkatkan kinerja dan mengatasi masalah tersebut, metode seperti pruning dan ensemble learning sering digunakan. Parameter max_depth menentukan kedalaman maksimum dari setiap decision tree dalam ensemble. Dalam kasus ini, decision tree dibatasi hingga kedalaman 8, yang membantu mencegah model dari overfitting terhadap data pelatihan dengan membatasi kompleksitas pohon. Parameter min_samples_split menentukan jumlah minimum sampel yang diperlukan untuk membagi node internal. Dengan nilai 20, setiap node internal harus memiliki setidaknya 20 sampel untuk membaginya menjadi dua cabang anak. Hal ii membantu mengontrol kompleksitas pohon dengan memastikan bahwa pemisahan hanya terjadi ketika cukup banyak sampel tersedia, sehingga menghasilkan pemisahan yang lebih signifikan dan mencegah pohon dari memecah terlalu dalam.

2. Random Forest

    Random forest adalah metode ensemble learning yang menggunakan banyak decision tree yang dibangun secara acak dari sampel data latih. Setiap pohon menghasilkan prediksi, dan prediksi akhir diambil berdasarkan mayoritas suara atau rata-rata. Keunggulan random forest termasuk kemampuannya mengatasi overfitting dan menangani data tidak seimbang. Metode ini cocok untuk berbagai masalah prediksi dan sering dipilih karena kinerja yang stabil dan akurat. Sama seperti metode Decision Tree, Random forest menggunakan parameter max_depth = 8 dan min_samples_split = 20

3. XGBoost

    XGBoost adalah algoritma machine learning yang sangat efektif dalam mengatasi berbagai jenis masalah prediksi, seperti klasifikasi dan regresi. Dengan menggunakan teknik ensemble boosting, XGBoost menggabungkan prediksi dari beberapa model lemah untuk membentuk model yang kuat. Keunggulannya termasuk kecepatan pelatihan yang tinggi, kemampuan menangani data tidak seimbang, dan penanganan fitur yang kuat. XGBoost yang digunakan menggunakan parameter alpha=2. Parameter alpha pada XGBoost adalah parameter yang mengontrol regularisasi L1 (atau juga dikenal sebagai regularisasi Lasso) pada model. Regularisasi L1 ditambahkan ke fungsi tujuan (objective function) sebagai bagian dari upaya untuk mencegah overfitting. Nilai alpha yang lebih besar akan memberikan penalti yang lebih besar terhadap bobot yang lebih besar, sehingga mendorong model untuk menjadi lebih sederhana dengan mengurangi bobot yang tidak signifikan. Dengan mengatur nilai alpha yang tepat, kita dapat mengendalikan kompleksitas model dan meningkatkan generalisasi pada data baru. Alpha=2 didapatkan dengan cara bereksperimen menggunakan alpha dari 1-10 dan mengambil hasil yang terbaik

Tahapan yang dilakukan adalah:

1. Mendefinisikan metode yang digunakan
2. Melakukan pelatihan model menggunakan dataset training
3. Melakukan testing pada dataset test untuk melihat performa model terhadap data baru
4. Membandingkan kinerja ketiga model menggunakan metrik akurasi dan MSE

## Evaluation
Pada kasus ini, digunakan metrik akurasi dan MSE untuk mengukur performa model.

Metrik akurasi mengukur seberapa sering model melakukan prediksi yang benar dari semua prediksinya. Secara matematis, akurasi dihitung dengan membagi jumlah prediksi yang benar dengan total jumlah prediksi yang dilakukan. Metrik ini cocok digunakan untuk masalah klasifikasi di mana outputnya adalah label atau kategori. Namun, perlu diingat bahwa akurasi tidak memberikan informasi tentang seberapa jauh prediksi yang salah dari nilai yang sebenarnya.

Mean Squared Error (MSE) adalah salah satu metrik evaluasi yang umum digunakan dalam masalah regresi dalam machine learning. MSE mengukur rata-rata dari kuadrat perbedaan antara nilai prediksi yang dihasilkan oleh model dengan nilai sebenarnya dari data yang diamati. Secara matematis, MSE dihitung dengan menjumlahkan kuadrat dari selisih antara nilai prediksi dan nilai sebenarnya untuk setiap observasi, kemudian membaginya dengan jumlah total observasi. MSE memberikan gambaran tentang seberapa dekat prediksi model dengan nilai sebenarnya, dengan nilai MSE yang lebih rendah menunjukkan bahwa model memiliki tingkat kesalahan yang lebih rendah dan memberikan prediksi yang lebih baik.

Berikut merupakan tabel perbandingan akurasi ketiga model:

|Model| Train Accuracy | Test Accuracy  |
|--------------|----------------|----------------|
| Decision Tree|     0.866032   |     0.865914   |
| Random Forest|     0.865529   |     0.865736   |
| XGBoost      |     0.874557   |     0.865618   |


Berikut merupakan tabel perbandingan MSE ketiga model:

|     Model| Train MSE       | Test MSE       |
|--------------|-----------------|----------------|
| Decision Tree|     0.000134    |     0.000134   |
| Random Forest|     0.000134    |     0.000134   |
| XGBoost      |     0.000125    |     0.00013    |

Hasil akurasi dan MAE (Mean Absolute Error) dari model-model yang dievaluasi menunjukkan performa yang serupa antara data training dan data testing. Dapat dilihat bahwa model Decision Tree, Random Forest, dan XGBoost memiliki tingkat akurasi yang relatif stabil antara data training dan data testing, dengan perbedaan yang sangat kecil. Hal ini menunjukkan bahwa model-model tersebut mampu melakukan prediksi dengan konsisten dan tidak terlalu memperlihatkan overfitting atau underfitting pada data.

Secara khusus, nilai akurasi untuk ketiga model berada di sekitar 86.5% menunjukkan bahwa ketiga model memiliki performa serupa dengan sebagian besar prediksi yang dilakukan oleh model benar sesuai dengan data sebenarnya. Selain itu, nilai MAE yang sangat kecil, sekitar 0.00013, menunjukkan bahwa kesalahan rata-rata dari prediksi model sangat rendah. Hal ini mengindikasikan bahwa model-model tersebut mampu melakukan prediksi dengan tingkat kesalahan yang minim dalam mengestimasi nilai target.

Meskipun metode decision tree memiliki akurasi terbaik, tetapi selisihnya sangat kecil dibandingkan dengan kedua metode lainnya
