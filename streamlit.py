import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Menghubungkan Google Drive dan membaca file CSV
def load_data():
    df = pd.read_csv('/content/drive/MyDrive/Project Pembelajaran Mesin/Salinan kc_house_data.csv', 
                     usecols=['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'price', 'yr_built'])
    return df

# Fungsi untuk Menampilkan Statistical Info
def display_statistics(df):
    st.subheader('Statistical Description')
    st.write(df.describe())
    st.subheader('Missing Values')
    st.write(df.isnull().sum())

# Fungsi untuk Menampilkan Visualisasi
def display_plots(df):
    st.subheader('Visualisasi Data')
    st.write("Distribusi dari Bedrooms")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(df['bedrooms'], ax=axes[0])
    axes[1].boxplot(df['bedrooms'])
    st.pyplot(fig)

    st.write("Distribusi dari Sqft Living")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df['sqft_living'].plot(kind='kde', ax=axes[0])
    axes[1].boxplot(df['sqft_living'])
    st.pyplot(fig)

    st.write("Distribusi dari Grade")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(df['grade'], ax=axes[0])
    axes[1].boxplot(df['grade'])
    st.pyplot(fig)

    st.write("Distribusi dari Year Built")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.countplot(df['yr_built'], ax=axes[0])
    axes[1].boxplot(df['yr_built'])
    st.pyplot(fig)

# Fungsi untuk Melatih dan Memprediksi Menggunakan Model Linear Regresi
def linear_regression_model(df):
    st.subheader('Linear Regression Model')
    
    # Menyiapkan data untuk pelatihan
    x = df.drop(columns='price')
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
    
    # Training model Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)

    # Menampilkan koefisien dan intercept
    st.write("Koefisien Linear Regression", lin_reg.coef_)
    st.write("Intercept Linear Regression", lin_reg.intercept_)

    # Menghitung akurasi
    score = lin_reg.score(x_test, y_test)
    st.write(f"Akurasi model Linear Regression: {score:.2f}")

# Fungsi untuk Melatih dan Memprediksi Menggunakan SVR
def svr_model(df):
    st.subheader('SVR Model')
    
    x = df.drop(columns='price')
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

    # Menangani data yang hilang
    x_train = x_train.fillna(x_train.mean())
    x_test = x_test.fillna(x_test.mean())

    # Standarisasi data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Melatih model SVR
    svr = SVR(kernel='rbf', C=100, epsilon=0.1)
    svr.fit(x_train_scaled, y_train)

    # Evaluasi model
    y_pred = svr.predict(x_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")
    
    return svr, scaler

# Fungsi untuk Prediksi Harga Rumah berdasarkan Input Pengguna
def predict_house_price(svr, scaler):
    st.subheader('Prediksi Harga Rumah')
    
    # Mengambil input dari pengguna
    bedrooms = st.number_input('Jumlah Kamar Tidur', min_value=1, max_value=10, value=3)
    bathrooms = st.number_input('Jumlah Kamar Mandi', min_value=1, max_value=10, value=2)
    sqft_living = st.number_input('Luas Rumah (sqft)', min_value=500, max_value=10000, value=2000)
    grade = st.number_input('Grade Rumah', min_value=1, max_value=13, value=7)
    yr_built = st.number_input('Tahun Dibangun', min_value=1900, max_value=2024, value=1990)

    # Input rumah Joko
    user_input = [[bedrooms, bathrooms, sqft_living, grade, yr_built]]
    
    # Standarisasi input pengguna
    user_input_scaled = scaler.transform(user_input)
    
    # Prediksi harga rumah
    predicted_price = svr.predict(user_input_scaled)
    st.write(f"Prediksi Harga Rumah: ${predicted_price[0]:,.2f}")

# UI Streamlit
def main():
    st.title('Prediksi Harga Rumah dengan Mesin Pembelajaran')

    # Muat data
    df = load_data()

    # Menampilkan data dan informasi
    display_statistics(df)

    # Menampilkan visualisasi
    display_plots(df)

    # Model Linear Regression
    linear_regression_model(df)

    # Model SVR
    svr, scaler = svr_model(df)

    # Prediksi harga rumah berdasarkan input pengguna
    predict_house_price(svr, scaler)

if __name__ == '__main__':
    main()

