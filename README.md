![image](https://github.com/ChandaniNovita/german_credit/assets/146313340/92aaa52e-0e67-4261-b40b-909287636123)# Laporan Proyek Machine Learning
### Nama : Chandani Novitasari
### Nim : 211351036
### Kelas : Malam A

## Domain Proyek

Dalam kumpulan data ini, setiap entri mewakili orang yang mengambil kredit dari bank. Setiap orang diklasifikasikan menjadi risiko kredit baik atau buruk menurut serangkaian atributnya. Pengklasteran k rata-rata adalah algoritme untuk membagi n pengamatan menjadi k kelompok sedemikian hingga tiap pengamatan termasuk ke dalam kelompok dengan rata-rata terdekat. Hasilnya adalah pembagian pengamatan ke dalam sel-sel Voronoi. Pengklasteran k rata-rata meminimalkan ragam dalam klaster.


## Business Understanding

Pengklusteran (clustering) dataset German Credit dapat memberikan pemahaman yang lebih baik tentang karakteristik dan pola yang mungkin ada dalam data kredit tersebut. 

### Problem Statements

Pernyataan Masalah 1:
Pengelolaan risiko kredit di belum optimal, menyebabkan tingginya tingkat keterlambatan pembayaran dan peningkatan risiko kredit. Kami perlu memahami karakteristik kelompok pelanggan yang berisiko tinggi dan mengidentifikasi strategi mitigasi risiko.

Pernyataan Masalah 2:
Proses penilaian kredit saat ini tidak efisien dan kurang mempertimbangkan faktor-faktor yang dapat mempengaruhi kemampuan pelanggan untuk membayar kredit. Kami menghadapi tantangan dalam memprediksi risiko kredit dengan tepat dan memerlukan perbaikan pada model penilaian kredit kami.

Pernyataan Masalah 3:
Analisis pelanggan kami saat ini belum memberikan wawasan yang memadai tentang preferensi dan perilaku kredit mereka. Kami perlu mengidentifikasi pola perilaku kredit yang dapat digunakan untuk menyusun strategi pemasaran yang lebih efektif dan meningkatkan pengalaman pelanggan.
### Goals

Jawaban Pernyataan Masalah 1:
Mengurangi tingkat risiko kredit dengan mengidentifikasi dan mengelompokkan pelanggan berdasarkan karakteristik kredit, sehingga memungkinkan pengembangan strategi risiko yang lebih cermat dan efektif.

Jawaban Pernyataan Masalah 2:
Meningkatkan efisiensi proses penilaian kredit dengan mengimplementasikan model penilaian yang lebih akurat dan dapat mempertimbangkan variabel-variabel yang relevan dalam menilai kemampuan pelanggan untuk membayar kredit.

Jawaban Pernyataan Masalah 3:
Memahami dengan lebih baik preferensi dan perilaku kredit pelanggan melalui analisis yang mendalam. Tujuannya adalah untuk menyusun strategi pemasaran yang lebih terarah dan meningkatkan kepuasan pelanggan melalui personalisasi layanan kredit.



## Data Understanding
Dataset yang saya gunakan berasal dari  [kaggle](https://www.kaggle.com/) dataset ini memiliki 1000 baris dan 9 kolom. Untuk lebih jelasnya klik tautan di bawah ini untuk melihat dataset yang saya gunakan.[German Credit Risk](https://www.kaggle.com/datasets/uciml/german-credit)

 

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:
- Age : Merupakan usia dari setiap orang yang ada di dalam data (numeric)
- Sex : Jenis kelamin(text: male, female)
- Job : Pekerjaan(numeric: 0 - unskilled and non-resident, 1 - unskilled and -resident, 2 - skilled, 3 - highly skilled)
- Housing : jenis rumah yang ditinggali(text: own, rent, or free)
- Saving accounts : Rekening tabungan untuk melihat keuangan (text - little, moderate, quite rich, rich)
- Checking account : cara pembayaran yang hampir merupakan kebalikan dari sistem cek, berupa surat perintah untuk memindahbukukan sejumlah uang dari rekening seseorang kepada rekening lain yang ditunjuk surat tersebut. (numeric, in DM - Deutsch Mark)
- Credit amount : jumlahk kredit (numeric, in DM)
- Duration : jangka waktu pinjaman atau kredit yang diberikan kepada debitur (peminjam) untuk membayarkan angsuran pinjaman yang diberikan oleh peminjam.(numeric, in month)
- Purpose : tujuan (text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others)


## Data Preparation
#### Import dataset
Pertama yang harus kita lakukan adalah mengimport dataset  dari kaggle
```python
from google.colab import files
files.upload()
```
```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```
Gunakan script berikut untuk mendownload datasetnya
```python
!kaggle datasets download -d uciml/german-credit
```
Ekstract file yang sudah di download
```python
!mkdir german-credit
!unzip german-credit.zip -d german-credit
!ls german-credit
```
Import dataset sudah selesai, selanjutnya kita akan melanjutkan ke proses berikut.
#### Import Library
Berikut adalah beberapa library yang akan kita gunakan.
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly as py
import os
import plotly.io as pio
pio.renderers.default='notebook'

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import scipy.cluster.hierarchy as shc
import plotly.graph_objects as go
```
Masukkan datasetnya dan melihat 5 isi pertama dari dataset tersebut.
```python
df =  pd.read_csv('german-credit/german_credit_data.csv', index_col=0)
df.head(5)
```
#### Data Discovery
Data discovery merujuk pada proses eksplorasi dan pemahaman data yang bertujuan untuk menemukan pola, tren, dan wawasan yang tersembunyi dalam dataset. Ini merupakan bagian penting dari analisis data dan memungkinkan pengguna untuk membuat keputusan berdasarkan pemahaman yang lebih baik tentang informasi yang terkandung dalam data.

Pada dataset ini didapatkan :

Terdapat 1000 row data dengan 9 kolom
Masih ada data yang kosong yaitu pada kolom saving accounts dan checking accounts
Terdapat 4 kolom dengan tipe integer dan 5 kolom dengan tipe data object
```python
df.info()
```
```python
df.describe(include=['object'])
```
Melihat nilai unik pada kolom yang bertipe kategori
```python
print('Nilai Unik Sex/Gender:', df['Sex'].unique())
print('')
print('Nilai Unik Housing/Status Tinggal:', df['Housing'].unique())
print('')
print('Nilai Unik Saving accounts/Uang Tabungan:', df['Saving accounts'].unique())
print('')
print('Nilai Unik Checking account/Tabungan Giro:', df['Checking account'].unique())
print('')
print('Nilai Unik Checking Purpose/Tujuan', df['Purpose'].unique())
```
Membedakan kolom yang bertipe numerik dan kategori
```python
numeric = ['Age', 'Job', 'Credit amount', 'Duration']
categorical = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
```
#### Exploratori Data Analysis(EDA)
Cek kembali data yang kosong

Didapatkan pada kolom Saving account 183 data kosong dan pada kolom Checking amount 394 data kossong sedangakan pada kolom lainnya tidak terdapat data yang kosong.
```python
df.isnull().sum()
```
Melihat lagi data unik pada kolom saving accounts dan checking account karena terdapat data yang kosong pada ke-2 kolom tsb.
```python
print('Data Uik saving accounts:', df['Saving accounts'].unique())
print('Data Unik checking account:', df['Checking account'].unique())
```
Mengisi data yang kosong tersebut dengan label 'unknown' atau tidak diketahui
```python
df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
df['Checking account'] = df['Checking account'].fillna('unknown')
df.head(5)
```
Visualisasi
```python
df.hist(figsize = (20,15));
```
![image](https://github.com/ChandaniNovita/german_credit/blob/main/visualisasi1.png)
Pada visualisasi pertama ini membuat visualisasi grafik pada data numerik yaitu kolom age, job, credit amount dan duration. Dan didapatkan :

Pada grafik age/umur terjadi pelonjakan kenaikan dari umur 20 ke 25 dan dari 25 sampai 40 merupakan umur yang dengan jumlah terbanyak dan semakin tua umurnya semakin menurun grafik
Pada grafik job label 2 mendominasi dengan jumlah >600
Pada grafik Credit amount paling banyak yaitu pada jumlah credit amount 0-3000 dan untuk semakin besar credit amount semakin sedikit jumlahnya juga
Pada grafik duration didapatkan durasi yang paling banyak diambil oleh nasabah yaitu dengan durasi 2-25 bulan, untuk yang lebih dari itu juga ada tetapi terdapat perbedaan grafik yang cukup berbeda signifikan lebih sedikit.
```python
for col in df[categorical].columns:
    sns.countplot(y =col, data = df)
    plt.show()
```
![image](https://github.com/ChandaniNovita/german_credit/blob/main/visualisasi2.png)
![image](https://github.com/ChandaniNovita/german_credit/blob/main/visualisasi4.png)
![image](https://github.com/ChandaniNovita/german_credit/blob/main/visualisasi5.png)
![image](https://github.com/ChandaniNovita/german_credit/blob/main/visualisasi6.png).
Pada visualisasi pertama ini membuat visualisasi grafik pada data categorical yaitu kolom sex, housing, saving accounts, checking account dan purpose. Dan didapatkan :

- Pada atribut Sex/Gender : Nasabah kredit bank lebih banyak dari gender laki-laki perbandingannya mencapai 7:3.
Pada atribut Housing/Rumah : Nasabah kredit bank paling banyak sudah memiliki rumah sendiri dan perbandingannya cukup dominan 7:1:2 (milik:bebas:sewa).
- Pada atribut Saving accounts/Jumlah tabungan : Nasabah kredit bank didominasi yang memiliki jumlah tabungan dengan jumlah sedikit.
- Pada atribut Checking accounts/Jumlah tabungan Giro: Nasabah kredit bank didominasi dengan nasabah yang tidak diketahui statusnya.
- Pada atribut Purpose : Nasabah kredit bank didominasi yang memiliki tujuan kredit yaitu untuk membeli mobil dan 3 tujuan tertinggi dibawahnya yaitu : radio/TV, furnitur/perlengkapan dan bisnis.

Pada proses ini saya menggunakan tabel grafik korelasi matriks unutk mengetahui korelasi antara tiap kolom/variable fitur. Terdapat kontras warna juga untuk mengetahui korelasi jika semakin berkorelasi warna nya akan semakin terang.
```python
corr = df.corr()
plt.figure(figsize=(10,8));
sns.heatmap(corr, annot=True, fmt='.2f');
```
![image](https://github.com/ChandaniNovita/german_credit/blob/main/visualisasi3.png)
#### Preprocessing
Menyalin data
```python
data = df.copy()
```
Pada proses ini dilakukan encode data yang mempunyai tipe data categorical/object menjadi integer.
```python
encoder = LabelEncoder()
from sklearn.preprocessing import LabelEncoder
for label in categorical:
    data[label] = encoder.fit_transform(data[label])
```
Menampilkan data kategori yang sudah diubah
```python
data[categorical]
```
Proses ini melakukan normalisasi data menggunakan Standard Scaler
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(X_scaled, columns=data.columns)
data_scaled.head()
```
```python
X = data
```
## Modeling
Model K-Means adalah algoritma klustering yang digunakan untuk mengelompokkan data menjadi beberapa kelompok atau kluster berdasarkan kesamaan karakteristik. Tujuan utama dari algoritma K-Means adalah meminimalkan varians intra-kluster, yaitu memastikan bahwa data di dalam satu kluster memiliki kemiripan yang tinggi.

Pada proses ini dilakukan klasterisasi data.
```python
clusters=[]
for i in range(1,11):
  km =KMeans(n_clusters=i).fit(X)
  clusters.append(km.inertia_)

fig, ax = plt.subplots(figsize=(12,8))
sns.lineplot(x=list(range(1,11)),y=clusters, ax=ax)
ax.set_title('Mencari elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('inertia')

```
![image](https://github.com/ChandaniNovita/german_credit/blob/main/elbow.png).
Pada proses ini mendefinisikan algoritma Machine Learning K-Means dan menyimpan hasil kuslter kedalam kolom baru yaitu kolom 'Cluster'
```python
def k_means(n_clust):
    X_copy = X.copy()
    kmean = KMeans(n_clusters=n_clust).fit(X_copy)
    X_copy['Labels'] = kmean.labels_
    return X_copy
```
####Visualisasi Hasil Algoritma
Berikut adalah grafik clustering 
```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X['Age'], X['Job'], X['Credit amount'], c=X['Labels'], cmap='viridis')

ax.set_xlabel('Age')
ax.set_ylabel('Job')
ax.set_zlabel('Credit amount')

legend = ax.legend(*scatter.legend_elements(), title='Labels')
ax.add_artist(legend)

plt.show()
```
![image](https://github.com/ChandaniNovita/german_credit/blob/main/cluster.png)
## Evaluation
Dengan menggunakan algoritma k-means dan konsep clustering, kita dapat mengelompokkan data menjadi kelompok-kelompok yang memiliki karakteristik serupa, membantu dalam pemahaman dan pengambilan keputusan lebih lanjut.

## Deployment
Berikut adalah link streamlitnya [German Credit](https://germancredit-8szqub5anhgtxhrwaestpv.streamlit.app/)
![image](https://github.com/ChandaniNovita/german_credit/blob/main/Screenshot%20(257).png)

