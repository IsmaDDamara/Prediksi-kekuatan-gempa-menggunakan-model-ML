# Prediksi Kekuatan Gempa Menggunakan Analisis Time Series dan Model Machine Learning

## Domain Proyek

Indonesia merupakan negara yang berada pada wilayah Cincin Api Aktif (<em>Ring of Fire</em>) yang menjadi pertemuan lempeng tektonik, sehingga menjadikan wilayah indonesia sering mengalami gempa bumi. Menurut Data yang di rilis oleh Pusat Badan Statistik, yang bersumber dari Badan *Meteorologi Klimatologi dan Geofisika* (<em> BMKG</em>) mengungkapkan bahwa terdapat 10.843 aktivitas gempa yang tercatat sepanjang tahun 2022. Sebanyak 217 di antaranya memiliki kekuatan di atas 5 magnitudo, tentu saja kerusakan yang ditimbulkan dari bencana gempa bumi ini sangatlah besar.[Sepanjang 2022, 217 Gempa Tektonik Guncang Indonesia](https://www.esdm.go.id/id/media-center/arsip-berita/sepanjang-2022-217-gempa-tektonik-guncang-indonesia#:~:text=Di%20tahun%202022%2C%20ada%20217,di%20Cianjur%22%2C%20terang%20Arifin.)

Berdasarkan data aktivitas gempa tersebut frekuensi gempa dalam satu tahun sudah sangat banyak dan kekuatannya pun banyak yang melebihi 5 magnitudo, sehingga tidak mengherankan gempa bumi merupakan salah satu bencana alam yang paling sulit untuk di prediksi. Lembaga pemerintah yang fokus menangani masalah bencana alam di indonesia pun merasa kesulitan dalam melakukan pencegahan dan penanggulangan bencana ini, dikarenakan gempa bumi merupakan bencana yang sulit untuk di prediksi karena aktivitas gempa bumi tidak menunjukkan suatu pola atau tren, kejadiannya pun terkadang tiba-tiba sehingga menimbulkan dampak kerugian yang sangat besar bukan hanya harta benda bahkan sudah ribuan nyawa yang melayang akibat bencana gempa bumi ini.

Oleh karena itu, dari permasalahan tersebut penulis ingin membuat sebuah model yang dapat memprediksi kekuatan gempa bumi di wilayah indonesia berdasarkan data-data yang sudah ada dari tahun ke tahun untuk membuat model yang tepat. Sehingga pemerintah dan masyarakat dapat dengan cepat melakukan langkah awal atau evakuasi untuk menyelamatkan diri mereka, sehingga jatuhnya korban jiwa dapat di kurangi.

## Business Understanding

### Problem Statements

Berdasarkan permasalahan yang sudah dijelaskan di atas <em>problem statements</em> dari proyek ini sebagai berikut::
-   Apakah terdapat pola atau kecenderungan dari kekuatan gempa bersamaan berjalannya waktu pada <em>Dataset</em> yang ada?
- Bagaimana cara melakukan proses <em>preprocessing</em>  yang efektif pada data yang diperoleh untuk membuat model <em> Time Series</em> yang akurat?
- Model <em>Machine Learning</em> manakah yang memiliki nilai <em>Root Mean Square Error (RMSE) </em>terkecil yang menandakan model semakin akurat?
- Seberapa akurat model untuk mengukur kekuatan gempa berdasarkan data yang telah ada?

### Goals 

Adapun tujuan yang hendak dicapai oleh penulis sebagai berikut:
- Melakukan visualisasi untuk melihat pola atau tren pada data
- Melakukan <em> preprocessing </em> untuk membersihkan data sehingga dapat menjadi data latih yang efektif untuk model <em>Time Series</em>
- Melakukan pengujian terhadap model <em>Machine Learning</em> yang sesuai dengan <em>Dataset</em> dan melihat model mana yang memiliki nilai <em>RMSE</em> terendah
- Melakukan visualisasi terhadap model yang telah dipilih untuk mengetahui seberapa akurat model untuk mengukur kekuatan gempa bumi berdasarkan data yang telah ada

### Solutions Statement


- Untuk melihat apakah terdapat pola atau kecenderungan pada <em>dataset time series</em> dapat dilakukan analisis tren menggunakan metode <em>rolling mean</em> sehingga jika terdapat pola pada data maka dapat di <em>visualisasikan</em> secara halus.
- Dalam proses <em>preprocessing </em> data dapat dilakukan beberapa proses di antaranya :
    - <em>Column Dropping</em> atau penghapusan kolom untuk menghapus kolom yang tidak digunakan selama proses pembuatan model atau kolom yang akan mengganggu proses analisis.
    - <em> Missing Data handling </em>atau penghapusan data yang hilang, bisa dengan cara menambahkan <em>mean</em> atau menghapus keseluruhan baris.
    - <em>Data Formating</em> untuk mengganti tipe data yang relevan untuk pembuatan model seperti mengganti tipe data menjadi <em>datetime </em>untuk memudahkan dalam analisis <em>Time Series</em>.
    - <em>Data Splitting</em> adalah proses untuk membagi data dengan rasio tertentu contohnya membagi data menjadi 80 : 20 untuk dijadikan sebagai data <em>train</em> dan data <em>test </em>.
    - Melakukan <em>Standarisasi</em> baik itu <em>Mix-Max Scalling atau Z Score</em> untuk menyeragamkan skala pada data.
- Untuk Pemilihan model terbaik untuk <em>Time Series</em> dapat dilakukan beberapa proses berikut :
    - Dalam pemilihan model untuk data <em>Time Series</em> kita tidak dapat melakukan teknik <em>lazy predict</em> untuk <em>evalusi</em> cepat dikarenakan teknik <em>lazy predict</em> mengesampingkan data waktu yang ada sehingga menghasilkan akurasi prediksi yang tidak akurat 
    - Untuk memilih model <em>Machine Learning</em> terbaik diharuskan untuk menguji secara langsung beberapa model yang biasa digunakan untuk analisis <em>time series</em>
    - Dalam proses uji model hitung <em>metric (rmse)</em> yang menjadi acuan memilih model terbaik
    - Proses penghitungan metric biasanya menggunakan library python yang bernama sklearn metrics
- Untuk melakukan visualisasi terhadap model yang telah dibuat guna melihat apakah prediksi model yang telah kita buat sesuai dengan data dapat dilakukan dengan melakukan proses visualisasi menggunakan library python yaitu <em>matplotlib</em>dan menggunakan diagram line chart untuk melihat pola data seiring waktu.

## Data Understanding
Dataset yang digunakan bersumber dari kaggle, dataset ini berisi informasi tanggal, tempat, kedalaman, dan kekuatan gempa yang terjadi dari tahun 2008 - awal tahun 2023 yang diambil oleh Badan *Meteorologi Klimatologi dan Geofisika* (<em> BMKG</em>), berikut adalah link datanya : https://www.kaggle.com/datasets/kekavigi/earthquakes-in-indonesia

### Variabel-variabel pada Dataset katalog_gempa adalah sebagai berikut:
- tgl : tanggal kejadian
- ot : stempel waktu acara
- lat : garis lintang episentrum peristiwa (derajat), mulai dari 6N hingga 11S
- lon : garis bujur pusat gempa (derajat), mulai dari 142E hingga 94E
- depth : kedalaman kejadian (km)
- mag : besarnya kekuatan gempa, mulai dari 1 hingga 9,5
- remark : Flinn-Engdahl wilayah acara terkadang, mekanisme fokus acara diukur. Dalam hal ini, nilai dip1, strike1, rake1, dip2, strike2, dan rake2 tidak kosong.

Data Informations :
    - Nama Dataset  : katalog_gempa
    - Jumlah Kolom  : 13
    - Jumlah Baris  : 92887
    - Missing Value : 0


### Analisis Deskriptif

|         |   count |       mean |        std |     min |     25% |    50% |    75% |    max |
| ------: | ------: | ---------: | ---------: | ------: | ------: | -----: | -----: | -----: |
|     lat | 92887.0 |  -3.404577 |   4.354584 |  -11.00 |  -7.885 |  -2.91 |   0.14 |   6.00 |
|     lon | 92887.0 | 119.159707 |  10.833202 |   94.02 | 113.170 | 121.16 | 126.90 | 142.00 |
|   depth | 92887.0 |  49.009399 |  76.761070 |    2.00 |  10.000 |  16.00 |  54.00 | 750.00 |
|     mag | 92887.0 |   3.592788 |   0.834042 |    1.00 |   3.000 |   3.50 |   4.20 |   7.90 |
| strike1 |  2735.0 | 170.142852 |  88.359267 |    0.00 | 107.550 | 144.60 | 217.50 | 359.20 |
|    dip1 |  2735.0 |  60.202121 |  19.699252 |    2.30 |  46.950 |  62.30 |  76.40 |  90.00 |
|   rake1 |  2735.0 |  30.358062 |  99.957906 | -180.00 | -28.500 |  57.60 | 100.15 | 180.00 |
| strike2 |  2735.0 | 197.450303 | 118.920519 |    0.00 |  63.115 | 240.72 | 297.48 | 359.98 |
|    dip2 |  2735.0 |  56.576344 |  21.274923 |    1.50 |  39.400 |  58.40 |  74.70 |  90.00 |
|   rake2 |  2735.0 |  35.250018 |  98.235894 | -180.00 | -19.900 |  56.50 | 112.60 | 180.00 |

<center>Tabel 1. <em>Deskripsi Statistik </em></center>

- Hasil Analisis
    - Terdapat 92887 jumlah data dengan 90152 nilai kosong pada kolom tertentu
    - Rata rata kekuatan gempa berada pada 3.5 magnitudo
    - Rata rata kedalaman gempa terjadi pada 49 kilometer di bawah laut
    - Kekuatan gempa bumi tertinggi yang tercatat pada periode 2008 - 2023 sebesar 7.9 magnitudo

### <em> Visualization </em>

Berikut adalah visualisasi untuk melihat karakteristik hubungan kolom tanggal dengan kolom magnitudo. Terlihat di sini bahwa pada tahun 2022 akhir kekuatan gempa bumi mengalami penurunan dengan kisaran 1 - 3 magnitudo tetapi juga ada beberapa titik pada tahun 2022 yang memiliki kekuatan gempa di atas 6 magnitudo.

Dari hasil visualisasi ini juga terlihat 5 tahun terakhir semakin banyak garis yang mengindikasikan banyaknya aktivitas gempa yang terjadi.

![corr](https://firebasestorage.googleapis.com/v0/b/gempa-bumi-ea3b1.appspot.com/o/visualisasi%20hubungan%20tanggal%20dan%20magnitudo.png?alt=media&token=c7c35272-b73b-4f88-9ad7-65b1a550bbda)

<center>Gambar 1. Grafik kekuatan gempa dari tahun 2008 - 2022</center>

### Analisis Tren

Analisis tren ini dilakukan untuk melihat pola atau kecenderungan kekuatan gempa seiring waktu untuk memahami peningkatan atau penurunan kekuatan gempa setiap tahunnya. Metode yang digunakan dalam proses ini adalah <em>rolling mean</em> yaitu metode penghalusan data untuk mengurangi fluktuasi dengan cara menghitung nilai rata-rata suatu jendela yang digeser sepanjang deret waktu.

Terlihat pada line berwarna merah merupakan nilai rata-rata <em>rolling meannya</em> yang membentuk deret waktu baru yang lebih halus.

![Tren](https://firebasestorage.googleapis.com/v0/b/gempa-bumi-ea3b1.appspot.com/o/analisis%20tren.png?alt=media&token=efbb588e-ac1a-4b90-b2b2-e9c9ae69280e)



<center>Gambar 2. Analisis Tren</center>

### Analisis Fluktuasi

Analisis fluktuasi ini dilakukan untuk  melihat suatu pola dalam data <em>time series</em> untuk memahami data secara mendalam sehingga dapat digunakan untuk mengambil keputusan yang lebih baik. Metode yang digunakan dalam proses analisis fluktuasi sebenarnya ada banyak, dalam kasus ini menggunakan metode <em>rolling standard deviation</em> untuk untuk menghitung simpanan baku atau <em>standard deviation</em> dari data.

![fluktuasi](https://firebasestorage.googleapis.com/v0/b/gempa-bumi-ea3b1.appspot.com/o/analisis%20fluktuasi.png?alt=media&token=0c6f3530-8b30-4c7e-b7ae-10d75dd40d0f)

<center>Gambar 3. Analisis Fluktuasi</center>

### Identifikasi Anomali 

Proses identifikasi anomali dilakukan untuk mengecek apakah terdapat data Outlier atau data yang extrim, biasanya pada analisis lain data outlier akan dihilangkan sedangkan pada analisis time series data outlier justru menjadi informasi yang berharga untuk pembuatan model.

Dalam kasus ini data outlier ditandai dengan titik warna merah yang mengindikasikan bahwa terdapat anomali berupa nilai magnitudo yang terlalu tinggi atau magnitudo yang terlalu kecil dari data lainnya.

![anomali](https://firebasestorage.googleapis.com/v0/b/gempa-bumi-ea3b1.appspot.com/o/analisis%20anomali.png?alt=media&token=879a279e-ee38-44a7-a9a7-0a005c29997c)

<center>Gambar 4. Identifikasi Anomali</center>

### Uji Stasioneritas

Proses uji stasioneritas dilakukan untuk mengetahui apakah data memiliki perubahan statistik setiap waktu, metode yang digunakan dalam proses uji stasioneritas adalah <em>Augmented Dickey-Fuller (ADF)</em> test.

> ADF Statistic: -18.93428068081308 
>
> p-value: 0.0 
>
> Critical Values: 
>
> 1% : -3.43042045335977 
>
> 5% : -2.8615711392546848 
>
> 10% : -2.5667865743402594

Dari hasil pengujian di atas menghasilkan nilai ADF statistic = -18.93 yang mana ini lebih kecil dari Critical Values 1%, 5% dan 10% sehingga H0 ditolak. Dapat disimpulkan bahwa data deret waktu berbentuk stasioner dan siap digunakan dalam pemodelan.

### Analisis ACF dan PACF

Proses analisis <em>ACF</em> dan <em>PACF</em> dilakukan untuk melakukan pengukuran korelasi antara nilai dalam deret waktu dengan dirinya sendiri pada waktu sebelumnya.  <em>ACF</em> biasanya digunakan untuk menemukan pola dalam data dan memberitahu bagaimana antar periode waktu berkorelasi satu sama lain.

![AC](https://firebasestorage.googleapis.com/v0/b/gempa-bumi-ea3b1.appspot.com/o/ACF.png?alt=media&token=8bb9ef8b-194b-44e8-94e9-02b51779fb0c)

<center>Gambar 5. Autocorrelation</center>

![](https://firebasestorage.googleapis.com/v0/b/gempa-bumi-ea3b1.appspot.com/o/ACF%20dan%20PACF.png?alt=media&token=93a41c4f-4659-4c51-847a-5b630ef82449)

<center>Gambar 6. Partial Autocorrelation</center>

## Data Preparation

Dalam kasus ini dikarenakan data tidak memiliki nilai null dan setiap kolom dan baris datanya bagus maka data tidak memerlukan perbaikan atau perubahan yang banyak, teknik <em>preparation</em> yang dilakukan sebagai berikut:

- **Column Selection** : proses mengambil kolom yang akan berguna untuk pembuatan model dan membuang kolom yang tidak berguna atau yang nantinya akan mengganggu
- **Renaming Column** : Proses mengganti nama kolom sehingga mudah dalam melakukan pemanggilan saat pembuatan model
- **Data Transformation** : Proses mengubah atau memanipulasi data asli menjadi bentuk yang lebih sesuai yang berguna untuk analisis atau pembuatan model, dalam kasus ini dikarenakan data terlalu banyak jadi mengharuskan dilakukan pengelompokan data berdasarkan interval waktu dan mengisi nilainya dengan nilai rata-rata dari data yang sudah disatukan tersebut sehingga akan menghasilkan nilai baru yang lebih efektif dalam pembuatan model. Dalam proses ini juga dilakukan perubahan tipe data kolom tanggal menjadi bertipe datetime untuk keperluan pembuatan model <em>time series</em>.

Berikut adalah final data setelah dilakukan proses preparation

Result:

|           |  count |      mean |       std |      min |       25% |       50% |       75% |        max |
| --------: | -----: | --------: | --------: | -------: | --------: | --------: | --------: | ---------: |
| Kedalaman | 4412.0 | 56.090949 | 33.480295 | 7.500000 | 37.477922 | 50.245690 | 66.190625 | 650.000000 |
| Magnitudo | 4412.0 |  3.730139 |  0.393675 | 2.514545 |  3.470352 |  3.691667 |  3.940000 |   5.804762 |

<center>Tabel 2. Deskripsi <em>statistic</em> setelah dilakukan <em>preprocessing</em></center>

Melakukan TimeSeries Split untuk membagi deret waktu yang akan digunakan sebagai data train dan data test

```
from sklearn.model_selection import TimeSeriesSplit

# Melakukan split pada data time series
tscv = TimeSeriesSplit(n_splits=10)
```

## Modeling

Pada tahap modeling akan dilakukan pengujian terhadap beberapa model yang biasa digunakan dalam analisis <em>time series</em> untuk menentukan model mana yang memiliki nilai  <em>Root Mean Square Error (RMSE)</em> terkecil yang mengartikan model memiliki akurasi yang tinggi.

### Pengujian Model

Dari beberapa model time series yang ada, dipilih menguji 3 model yang paling terkenal yaitu :

- ARIMA (AutoRegressive Integrated Moving Average) adalah metode pemodelan statistik yang digunakan untuk menganalisis dan meramalkan deret waktu.
- EXPONENTIAL SMOOTHING adalah metode perataan yang digunakan dalam analisis deret waktu untuk meramalkan nilai masa depan berdasarkan data masa lalu. 
- SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables) adalah salah satu jenis model statistik yang digunakan untuk analisis deret waktu. Model ini merupakan pengembangan dari model ARIMA dengan tambahan variabel eksogen atau faktor luar.

Dalam pengujian ini kita akan melihat model mana yang memiliki RMSE (Root Mean Square Error) yang paling kecil, semakin kecil nilai RMSE maka semakin akurat hasil prediksi time seriesnya.

Kita diharuskan menginstall terlebih dahulu library python yang dibutuhkan dikarenakan di colab tidak menyediakan library nya dengan cara :

```
!pip install statsmodels
```

#### Model ARIMA

Melakukan pengujian model ARIMA menggunakan code berikut :

```
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# order disini merupakan parameter yang digunakan dalam model arima
order = (1, 1, 1)

# Perform Time Series Cross-validation
rmse_arima_scores = []

for train_index, test_index in tscv.split(df['Magnitudo']):
    train, test = df.iloc[train_index], df.iloc[test_index]

    # Melatih Model ARIMA
    model_arima = ARIMA(train['Magnitudo'], order=order)
    model_fit_arima = model_arima.fit()

    # membuat prediksi dari data set test
    predictions = model_fit_arima.predict(start=test.index[0], end=test.index[-1])

    # Evaluasi model dan hasil prediksi
    rmse_arima = np.sqrt(mean_squared_error(test['Magnitudo'], predictions))
    rmse_arima_scores.append(rmse_arima)

# Calculate the average RMSE
avg_rmse_arima = np.mean(rmse_arima_scores)

# Print the average RMSE
print("Average RMSE ARIMA:", avg_rmse_arima)
```

Penjelasan Code :

Pertama melakukan pembagian deret waktu menjadi 10 bagian menggunakan <em>time series Cross-validation</em>, tiap bagian digunakan untuk data uji dan bagian sisanya digunakan sebagai data latih untuk model ARIMA. Mengatur parameter nya yaitu p=1, d=1, dan q=1 untuk melatih model, hasil dari pengujian model adalah <em>Root Mean Squared Error(RMSE)</em> yang mengartikan semakin kecil nilainya maka semakin akurat model tersebut.

#### Model EXPONENTIAL SMOOTHING

Melakukan pengujian model EXPONENTIAL SMOOTHING menggunakan code berikut :

```
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Perform Time Series Cross-validation
rmse_es_scores = []

for train_index, test_index in tscv.split(df['Magnitudo']):
    train, test = df.iloc[train_index], df.iloc[test_index]

    # Membuat dan melatih model Exponential Smoothing
    seasonal_periods = 12  # Assuming monthly data (if your data spans multiple years)
    model_es = ExponentialSmoothing(train['Magnitudo'], seasonal='add', seasonal_periods=seasonal_periods)
    model_fit_es = model_es.fit()

    # Melakukan prediksi pada data test set
    forecast_values = model_fit_es.forecast(steps=len(test))

    # Evaluasi model dan hasil prediksi
    mse_es = ((forecast_values - test['Magnitudo']) ** 2).mean()
    rmse_es = np.sqrt(mse_es)
    rmse_es_scores.append(rmse_es)

# Calculate the average RMSE across all folds
avg_es_rmse = np.mean(rmse_es_scores)

# Print the average RMSE
print('RMSE Exponential Smoothing:', avg_es_rmse)
```

Penjelasan Code :

Pertama melakukan pembagian deret waktu menjadi 10 bagian menggunakan <em>time series Cross-validation <em>, tiap bagian digunakan untuk data uji dan bagian sisanya digunakan sebagai data latih untuk model ES. Model tersebut menggunakan metode 'additive' untuk komponen musiman, yang berarti bahwa pola musiman ditambahkan ke nilai dasar. Setelah melatih model Exponential Smoothing, dilakukan prediksi pada data uji dengan menghitung nilai forecast_values. hasil dari pengujian model adalah Root Mean Squared Error(RMSE) yang mengartikan semakin kecil nilainya maka semakin akurat model tersebut.

#### Model SARIMAX

Melakukan pengujian model EXPONENTIAL SMOOTHING menggunakan code berikut :

```
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Perform Time Series Cross-validation
rmse_sarimax_scores = []

for train_index, test_index in tscv.split(df['Magnitudo']):
    train, test = df.iloc[train_index], df.iloc[test_index]

    # Membuat dan melatih model SARIMAX
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)  # Assuming monthly data with yearly seasonality
    model_sarimax = SARIMAX(train['Magnitudo'], order=order, seasonal_order=seasonal_order)
    model_fit_sarimax = model_sarimax.fit()

    # Melakukan prediksi pada data test set
    forecast_values = model_fit_sarimax.forecast(steps=len(test))

    # Evaluasi model dan hasil prediksi
    mse_sarimax = ((forecast_values - test['Magnitudo']) ** 2).mean()
    rmse_sarimax = np.sqrt(mse_sarimax)
    rmse_sarimax_scores.append(rmse_sarimax)

# Calculate the average RMSE across all folds
avg_sarimax_rmse = np.mean(rmse_sarimax_scores)

# Print the average RMSE
print('RMSE SARIMAX:', avg_sarimax_rmse)
```

Penjelasan Code :

Pertama melakukan pembagian deret waktu menjadi 10 bagian menggunakan <em>time series Cross-validation</em>, tiap bagian digunakan untuk data uji dan bagian sisanya digunakan sebagai data latih untuk model <em>SARIMAX</em>. Mengatur parameternya yaitu p=1, d=1, dan q=1 untuk melatih model, Selain itu, model juga menggunakan parameter (1, 1, 1, 12) untuk komponen musiman,  hasil dari pengujian model adalah <em>Root Mean Squared Error(RMSE) </em>yang mengartikan semakin kecil nilainya maka semakin akurat model tersebut.

Dari hasil pengujian dapat dilihat bahwa model <em>arima</em> memiliki nilai <em>Root Mean Square Error (RMSE)</em> terkecil sehingga dapat disimpulkan model <em>arima</em> merupakan model yang paling akurat dari ke 3 model yang telah diuji.

|Model|RMSE       |
|------|-----------|
|Arima|0.340       |
|Exponential Smoothing|0.344       |
|Sarimax|0.355       |

<center>Tabel 3. Hasil pengujian model</center>

#### Pengujian Model ARIMA Terhadap Data

Untuk melihat apakah model arima sesuai terhadap data, lakukan pengujian terhadap data yang ada dan lihat hasil prediksinya, gunakan code berikut untuk melakukannya:

```
# Mengubah tipe data kolom tanggal
df['Tanggal'] = pd.to_datetime(df['Tanggal'])

# Define the ARIMA order
order = (1, 1, 1)

# Perform Time Series Cross-validation
tscv = TimeSeriesSplit(n_splits=10)
rmse_scores = []

for train_index, test_index in tscv.split(df['Magnitudo']):
    train, test = df.iloc[train_index], df.iloc[test_index]

    # Fit the ARIMA model
    model = ARIMA(train['Magnitudo'], order=order)
    model_fit = model.fit()

    # Make predictions
    predictions = model_fit.predict(start=test.index[0], end=test.index[-1])

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test['Magnitudo'], predictions))
    rmse_scores.append(rmse)

# Calculate the average RMSE
avg_rmse = np.mean(rmse_scores)

# Print the average RMSE
print("Average RMSE:", avg_rmse)

# Fit the ARIMA model using the entire data
model = ARIMA(df['Magnitudo'], order=order)
model_fit = model.fit()

# Make predictions for the entire data
forecast_values = model_fit.predict(start=df.index[0], end=df.index[-1])

# Plot the results
plt.plot(df['Tanggal'], df['Magnitudo'], label='Data Historis')
plt.plot(df['Tanggal'], forecast_values, label='Prediksi')
plt.xlabel('Tanggal')
plt.ylabel('Magnitudo')
plt.title('Prediksi Gempa Bumi dengan ARIMA')
plt.legend()
plt.show()
```

Berikut Hasilnya :

![Uji Arima](https://firebasestorage.googleapis.com/v0/b/proyek-akhir-ai.appspot.com/o/prediksi2.png?alt=media&token=2de78729-b76c-4af4-a39f-2f8cb98ad166)

<center>Gambar 7. Hasil Uji Model Arima</center>

#### Kelebihan dan Kekurangan Model

#### Parameters
- <em>p (Order of Autoregression) </em>: Menunjukkan jumlah lag dari variabel dependen yang akan digunakan dalam model.
- <em>d (Order of Difference) </em>: Menunjukkan seberapa sering diferensiasi akan diterapkan pada data untuk membuatnya menjadi stasioner. 
- <em>q (Order of Moving Average) </em>: Menunjukkan jumlah lag dari residual/error yang akan digunakan dalam model.<em> 
- <em>s (Seasonal Period) </em>: Menunjukkan periode musiman dalam data.

#### Models

- <em>Arima</em> = ARIMA adalah model yang menggambarkan hubungan antara nilai deret waktu saat ini dan nilai masa lalunya, serta istilah kesalahan acak. Ini memiliki tiga parameter: p, d, dan q. Parameter p mewakili jumlah nilai yang tertinggal atau istilah autoregresif. Parameter d mewakili tingkat perbedaan atau integrasi. Parameter q mewakili jumlah istilah rata-rata bergerak atau istilah kesalahan. ([Autoregressive Integrated Moving Average (ARIMA)](https://elibrary.unikom.ac.id/id/eprint/3595/8/UNIKOM_ANDRI%20HADIANSYAH_BAB%20II.pdf))
  
    - Kelebihan
        - Merupakan model tanpa teori karena variabel yang digunakan adalah nilai-nilai lampau dan kesalahan yang mengikutinya.
        - Memiliki tingkat akurasi peramalan yang cukup tinggi karena setelah mengalami pengukuran kesalahan peramalan MAE (mean absolute error), nilainya mendekati nol.
        - Cocok digunakan untuk meramal sejumlah variabel dengan cepat, sederhana, akurat dan murah karena hanya membutuhkan data variabel yang akan diramal.
        
    - Kekurangan
        - Untuk data peramalan dalam periode yang cukup panjang ketepatannya kurang baik karena biasanya akan cenderung flat (datar/konstan). 
        - ARIMA akan mengalami penurunan keakuratan apabila terdapat komponen nonlinier time series pada data pengamatan.
        
    - Parameter

        - p = 1 
        - d = 1
        - q = 1
    
    - Cara kerja 
      
        Terdapat 3 cara kerja utama yaitu:
        
        - Rumus Autoregressive (AR) (p) adalah sebagai berikut: 
          $$
          y_t = c + φ_1 * y_(t-1) + φ_2 * y_(t-2) + ... + φ_p * y_(t-p) + ε_t
          $$
        
        - Rumus Integrated (I) (d) adalah sebagai berikut:
        
          
          $$
          y_t^d = (y_t - y_(t-d))
          $$
        
        - Rumus Moving Average (MA) (q) adalah sebagai berikut:
          $$
          y_t = c + θ_1 * ε_(t-1) + θ_2 * ε_(t-2) + ... + θ_q * ε_(t-q) + ε_t
          $$
    
    
    
- <em>Exponential Smoothing</em> = Exponential smoothing atau dalam bahasa Indonesia disebut dengan Penghalusan Eksponensial adalah suatu metode peramalan rata-rata bergerak yang memberikan bobot secara eksponensial atau bertingkat pada data-data terbarunya sehingga data-data terbaru tersebut akan mendapatkan bobot yang lebih besar. 
  
  Kelebihan utama dari metode exponential smoothing adalah dilihat dari kemudahan dalam operasi yang relative rendah, ada sedikit keraguan apakah ketepatan yang lebih baik selalu dapat dicapai dengan menggunakan (QS) Quantitatif sistem ataukah metode dekonposisi yang secara intuitif menarik, namun dalam hal ini jika diperlukan peramalan untuk ratusan item.
  
  Kelemahan dari metode exponential smoothing adalah adanya tingkat keraguan yang tinggi apabila metode tersebut digunakan pada peramalan jangka panjang, penggunaan dalam peramalan harus benar-benar memperhatikan nilai konstanta pemulusan agar hasil peramalan dapat lebih akurat. ([Metode Exponential Smoothing](https://jawabanapapun.com/apa-kelebihan-utama-dari-metode-exponential-smoothing/))
  
    -  Cara Kerja
  
        - Rumus Single Exponential Smoothing sebagai berikut:
  
          
          $$
          Y^t+1=α⋅Yt+(1−α)⋅Y^t
          $$
  
        - Rumus Double Exponential Smoothing (Holt's method) sebagai berikut:
             $$
             Y^t+1=α⋅Yt+(1−α)⋅(Y^t+bt)
             $$
  
             $$
             bt+1=β⋅(Y^t+1−Y^t)+(1−β).bt
             $$
  
        - Rumus Triple Exponential Smoothing (Holt-Winters' method) sebagai berikut:
             $$
             Y^t+m=(Y^t+m⋅bt)⋅st−m+k
             $$
  
             $$
             bt+m=β⋅(Yt+m−Y^t)+(1−β)⋅bt
             $$
  
             $$
             st+m=γ⋅((Y^t+m⋅bt)Yt+m)+(1−γ)⋅st−m+k
             $$
  
    - Parameter
  
        - s = 12
  
- <em>Sarimax </em>= SARIMAX (Seasonal Autoregressive Integrated Moving Average with eXogenous) adalah model untuk meramalkan data berpola musiman dengan beberapa variabel independen yang mempengaruhinya. Konsep fungsi transfer multivariat diterapkan pada metode ini.Kelebihan dari metode SARIMAX adalah dapat memodelkan data dengan pola musiman dan dapat memasukkan variabel independen yang mempengaruhi data tersebut. 

  Sedangkan kekurangan dari metode SARIMAX adalah memerlukan data yang lengkap dan tidak boleh ada missing value pada data. Selain itu, metode ini juga memerlukan waktu komputasi yang cukup lama. ([Sarimax](http://repository.upi.edu/39338/))

  - Cara Kerja

    - Rumus sebagai berikut :
      $$
      Y(t) = AR(p) + MA(q) + SAR(P, s) + X + ε(t)
      $$

  - Parameter
      - p = 1
      - d = 1
      - q = 1
      - s = 12
## Evaluation

Model yang digunakan merupakan model jenis Timer Series sehingga membutuhkan metric sebagai berikut:
- Root Mean Squared Error (RMSE)

### Root Mean Squared Error (RMSE)
Root Mean Squared Error (RMSE) adalah metode evaluasi yang menggunakan dasar pada jumlah kesalahan kuadrat atau selisih antara nilai real atau nyata dengan nilai prediksi yang telah diatur.

Kelebihan dari Metode RMSE ini antara lain yaitu menghasilkan nilai absolut. RMSE juga digunakan ketika kesalahan kecil dan kesalahan besar perlu diabaikan dan dikurangi sebanyak mungkin.

Kekurangan RMSE sendiri yaitu lebih sulit dipahami dan memerlukan perhitungan tambahan karena menggunakan akar kuadrat.([RMSE](https://www.trivusi.web.id/2023/03/perbedaan-mae-mse-rmse-dan-mape.html#:~:text=Kelebihan%20dan%20Kekurangan%20RMSE%20Kelebihan%20dari%20RMSE%20adalah,dan%20memerlukan%20penghitungan%20tambahan%20karena%20melibatkan%20akar%20kuadrat.))

Rumus matematika untuk RMSE adalah:
$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$
di mana:

-  ( n ) adalah jumlah observasi dalam dataset.
- ( y_i ) adalah nilai sebenarnya pada observasi ke-i.
- ( y_i ) adalah nilai prediksi pada observasi ke-i.


### Final Report

Setelah dilakukan tahap evaluasi dapat diambil kesimpulan bahwa model terbaik yang akan digunakan adalah model ARIMA sesuai dengan hasil penghitungan menggunakan metrix RMSE. Berikut beberapa model yang telah diuji :

Tabel 5. <em> Final Result of Model </em>

| Model                 | RMSE  |
| --------------------- | ----- |
| Arima                 | 0.340 |
| Exponential Smoothing | 0.344 |
| Sarimax               | 0.355 |

<center>Tabel 4. Hasil evalusi model</center>

Hasil pengujian terhadap model arima berdasarkan data yang tersedia sangat memuaskan dimana model tersebut terbukti memiliki akurasi yang sangat tinggi sehingga dapat digunakan dalam proses prediksi kekuatan gempa bumi kedepannya.

## Daftar Referensi
Referensi

1. https://www.esdm.go.id/id/media-center/arsip-berita/sepanjang-2022-217-gempa-tektonik-guncang-indonesia#:~:text=Di%20tahun%202022%2C%20ada%20217,di%20Cianjur%22%2C%20terang%20Arifin.
2. https://elibrary.unikom.ac.id/id/eprint/3595/8/UNIKOM_ANDRI%20HADIANSYAH_BAB%20II.pdf
3. https://jawabanapapun.com/apa-kelebihan-utama-dari-metode-exponential-smoothing/
4. http://repository.upi.edu/39338/
5. https://www.trivusi.web.id/2023/03/perbedaan-mae-mse-rmse-dan-mape.html#:~:text=Kelebihan%20dan%20Kekurangan%20RMSE%20Kelebihan%20dari%20RMSE%20adalah,dan%20memerlukan%20penghitungan%20tambahan%20karena%20melibatkan%20akar%20kuadrat.



