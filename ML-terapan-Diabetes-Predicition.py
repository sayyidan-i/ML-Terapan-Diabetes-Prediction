# %% [markdown]
# # Prediksi Diabetes

# %% [markdown]
# ## Import Package

# %%
!pip install ucimlrepo

# %%
!pip install xgboost

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, roc_curve, confusion_matrix




# %% [markdown]
# ## Import Dataset

# %%
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
  
# data (as pandas dataframes) 
X = cdc_diabetes_health_indicators.data.features 
y = cdc_diabetes_health_indicators.data.targets 
  


# %%
df_features = pd.DataFrame(X)
df_features

# %%
df_label = pd.DataFrame(y)
df_label

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# Tahapan Data Preparation adalah sebagai berikut:
# 1. Cek informasi kolom
# 2. Mengatasi missing value
# 3. Memilih fitur yang akan dilatih
# 4. Melakukan normalisasi data
# 5. Melakukan split dataset 80:20 untuk data latih dan data uji

# %%
df_features.info()

# %%
df_label.info()

# %% [markdown]
# Berdasarkan informasi diatas, diketahui terdapat 20 kolom fitur dengan tipe data integer dan kolom label dengan tipe data integer pula

# %%
# Mengecek deskripsi data
df_features.describe()

# %%
# metadata 
print(cdc_diabetes_health_indicators.metadata) 
  
# variable information 
print(cdc_diabetes_health_indicators.variables)

# %% [markdown]
# Dari data tersebut, dapat dilihat bahwa:
# 1. terdapat 7 fitur dengan tipe integer, yaitu BMI, GenHlth, MentHlth, PhysHlth, Age, Education, Income
# 2. terdapat 15 fitur dengan tipe binary, yaitu HighBp, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthCare, NoDocbcCost, Diffwalk, dan Sex

# %% [markdown]
# ### Menangani Missing Value
# 

# %%
# Cek apakah terdapat missing value
df_features.isnull().sum()

# %%
df_label.isnull().sum()

# %% [markdown]
# Dapat dilihat bahwa dataset ini tidak memiliki missing value sehingga tidak diperlukan pra-pemrosesan untuk menanganinya

# %% [markdown]
# ### Memilih fitur untuk dilatih

# %%
# Confusion Matrix

df = pd.concat([df_features, df_label], axis=1)
corr_matrix = df.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='Set2', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# %% [markdown]
# Total terdapat 21 fitur, tetapi akan dipilih 15 fitur dengan korelasi tertinggi dengan harapan model menjadi lebih efisien

# %%
# Get absolute values of correlations
abs_correlations = corr_matrix['Diabetes_binary'].abs()

# Get the top 15 features with the highest correlations
top_15_features = abs_correlations.sort_values(ascending=False).index[1:16]

print(top_15_features)

# %%
print(corr_matrix['Diabetes_binary'].abs().sort_values(ascending=False))

# %%
df_top_features = df_features[top_15_features]
df_top_features

# %% [markdown]
# ### Melakukan Normalisasi Fitur

# %% [markdown]
# 
# Normalisasi fitur penting dalam pemrosesan data karena dapat meningkatkan kinerja model machine learning serta membantu dalam konvergensi algoritma pembelajaran. Dengan normalisasi, fitur-fitur dalam dataset diperkecil rentang nilainya ke dalam skala yang seragam, seperti antara 0 dan 1. Hal ini membantu dalam menghindari dominasi fitur dengan rentang nilai yang besar, yang dapat mengakibatkan model menjadi sensitif terhadap skala fitur dan performa yang buruk. Selain itu, normalisasi juga dapat membantu algoritma pembelajaran konvergen lebih cepat karena memperbaiki kondisi numerik dari data, seperti meningkatkan numerik kestabilan dan mengurangi kesalahan pembelajaran. Dengan demikian, normalisasi fitur merupakan langkah penting dalam persiapan data sebelum melatih model machine learning.

# %%
# Ambil kolom dengan nilai numerik selain binary

num_col =  ['GenHlth', 'BMI', 'Age','PhysHlth', 'Income', 'Education', 'MentHlth']

scaler = MinMaxScaler(feature_range=(0, 1))
df_top_features[num_col] = scaler.fit_transform(df_top_features[num_col])

# %%
df_top_features.head()

# %% [markdown]
# ### Split Dataset
# Pembagian data menjadi set pelatihan (training) dan validasi adalah praktik umum dalam pengembangan model machine learning. Data training digunakan untuk melatih model, sementara data validasi digunakan untuk mengevaluasi kinerja model di luar sampel data pelatihan. Proses ini penting untuk mengukur seberapa baik model akan berkinerja pada data baru yang belum pernah dilihat sebelumnya. Dengan memisahkan data menjadi kedua set ini, kita dapat memvalidasi keakuratan dan kemampuan generalisasi model secara objektif. Validasi juga membantu dalam mengidentifikasi apakah model cenderung overfitting (terlalu cocok dengan data pelatihan) atau underfitting (tidak cukup cocok dengan data pelatihan). Dengan demikian, pemisahan data menjadi set pelatihan dan validasi adalah langkah kritis dalam pengembangan model machine learning yang stabil dan berkinerja tinggi.

# %%
# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(df_top_features, df_label, test_size=0.2, random_state=42)

# %%
print(f'train samples: {len(X_train)}')
print(f'validation samples: {len(X_test)}')
print(f'train labels: {len(y_train)}')
print(f'validation labels: {len(y_test)}')

# %% [markdown]
# # Modelling

# %% [markdown]
# Model machine learning yang akan dibandingkan adalah Decision Tree, Random Forest, dan XGBoost

# %% [markdown]
# ### Decision Tree
# Decision tree adalah salah satu metode dalam machine learning yang digunakan untuk melakukan pemodelan prediktif dan mengambil keputusan berdasarkan serangkaian aturan yang dihasilkan dari data latih. Pada dasarnya, decision tree menggambarkan struktur pohon di mana setiap simpul internal mewakili keputusan berdasarkan fitur-fitur data, sedangkan cabang-cabangnya merepresentasikan hasil dari keputusan tersebut. Proses pembuatan decision tree melibatkan pemilihan fitur yang paling informatif untuk membagi data secara rekursif sehingga setiap cabang dari pohon meminimalkan ketidakmurnian (misclassification). Keuntungan utama dari decision tree adalah kemampuannya untuk menghasilkan aturan yang mudah diinterpretasi, karena representasi visualnya yang mirip dengan alur pemikiran manusia. Namun, decision tree cenderung rentan terhadap overfitting jika tidak diatur dengan baik, dan untuk meningkatkan kinerja dan mengatasi masalah tersebut, metode seperti pruning dan ensemble learning sering digunakan.

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, roc_curve, confusion_matrix



dt = DecisionTreeClassifier(max_depth=8, min_samples_split=20)

dt.fit(X_train, y_train)
predictions_train = dt.predict(X_train)
predictions_val = dt.predict(X_test)
accuracy_train = accuracy_score(predictions_train,y_train)
accuracy_val = accuracy_score(predictions_val,y_test)

print(f"accuracy_train: {accuracy_train} ")
print(f"accuracy_val: {accuracy_val} ")

print('\nClassification Report:')
print(classification_report(y_test,predictions_val))

sns.heatmap(confusion_matrix(y_test,predictions_val),fmt='',annot=True)


# %% [markdown]
# ### Random Forest
# Random forest adalah metode ensemble learning yang menggunakan banyak pohon keputusan yang dibangun secara acak dari sampel data latih. Setiap pohon menghasilkan prediksi, dan prediksi akhir diambil berdasarkan mayoritas suara atau rata-rata. Keunggulan random forest termasuk kemampuannya mengatasi overfitting dan menangani data tidak seimbang. Metode ini cocok untuk berbagai masalah prediksi dan sering dipilih karena kinerja yang stabil dan akurat.

# %%
from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier(max_depth=8, min_samples_split=20)

rf.fit(X_train, y_train)
predictions_train = rf.predict(X_train)
predictions_val = rf.predict(X_test)
accuracy_train = accuracy_score(predictions_train,y_train)
accuracy_val = accuracy_score(predictions_val,y_test)

print(f"accuracy_train: {accuracy_train} ")
print(f"accuracy_val: {accuracy_val} ")
print('\nClassification Report:')
print(classification_report(y_test,predictions_val))

sns.heatmap(confusion_matrix(y_test,predictions_val),fmt='',annot=True)


# %% [markdown]
# ### XGBoost
# XGBoost adalah algoritma machine learning yang sangat efektif dalam mengatasi berbagai jenis masalah prediksi, seperti klasifikasi dan regresi. Dengan menggunakan teknik ensemble boosting, XGBoost menggabungkan prediksi dari beberapa model lemah untuk membentuk model yang kuat. Keunggulannya termasuk kecepatan pelatihan yang tinggi, kemampuan menangani data tidak seimbang, dan penanganan fitur yang kuat.

# %%
from xgboost import XGBClassifier

xgb = XGBClassifier(alpha=2)

xgb.fit(X_train, y_train)
predictions_train = xgb.predict(X_train)
predictions_val = xgb.predict(X_test)
accuracy_train = accuracy_score(predictions_train,y_train)
accuracy_val = accuracy_score(predictions_val,y_test)

print(f"accuracy_train: {accuracy_train} ")
print(f"accuracy_val: {accuracy_val} ")


print('\nClassification Report:')
print(classification_report(y_test,predictions_val))

sns.heatmap(confusion_matrix(y_test,predictions_val),fmt='',annot=True)


# %%
#MSE
acc = pd.DataFrame(columns=['train', 'test'], index=['Decision Tree','Random Forest', 'XGBoost'])
mse = pd.DataFrame(columns=['train', 'test'], index=['Decision Tree','Random Forest', 'XGBoost'])
dict_model = {'Decision Tree': dt, 'Random Forest': rf, 'XGBoost': xgb}

for name, model in dict_model.items():
    acc.loc[name, 'train'] = accuracy_score(y_train, model.predict(X_train))
    acc.loc[name, 'test'] = accuracy_score(y_test, model.predict(X_test))
    
for name, model in dict_model.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3 
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3


# %%
acc

# %%
mse

# %%
import matplotlib.pyplot as plt

# Create subplots with 2 rows and 1 column
fig, axs = plt.subplots(2, 1)

# Plot Accuracy
acc.sort_values(by='test', ascending=False).plot(kind='barh', ax=axs[0], zorder=3)
axs[0].grid(zorder=0)
axs[0].set_title('Accuracy')

# Plot MSE
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=axs[1], zorder=3)
axs[1].grid(zorder=0)
axs[1].set_title('MSE')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# %% [markdown]
# ### Hasil
# Pada data validasi, ketiga model (decision tree, random forest, dan XGBoost) menunjukkan performa yang nyaris sama berdasarkan akurasi dan Mean Squared Error (MSE). Hal ini menunjukkan bahwa ketiga model memiliki kemampuan yang serupa dalam melakukan prediksi terhadap data baru yang belum pernah dilihat sebelumnya. Meskipun ketiga model menggunakan pendekatan yang berbeda dalam pembuatan prediksi (seperti single tree vs ensemble learning), namun hasil validasi menunjukkan bahwa mereka memberikan hasil yang sebanding dalam hal akurasi dan kesalahan prediksi. Keputusan ini dapat memberikan kepercayaan bahwa ketiga model tersebut dapat digunakan secara setara dalam aplikasi praktis, dengan mempertimbangkan faktor-faktor lain seperti kompleksitas model dan kebutuhan komputasi.

# %% [markdown]
# 


