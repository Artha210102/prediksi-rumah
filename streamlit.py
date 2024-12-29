import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Load the dataset
st.title("House Price Prediction App")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Select features and target
    if set(['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'price', 'yr_built']).issubset(df.columns):
        df = df[['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'price', 'yr_built']]

        # Display basic statistics
        st.write("### Dataset Info")
        st.write(df.describe())

        # Handle missing values
        df = df.fillna(df.mean())

        # Data Visualization
        st.write("### Data Visualization")
        f, axes = plt.subplots(2, 3, figsize=(15, 10))
        sns.histplot(df['bedrooms'], kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('Distribution of Bedrooms')
        sns.histplot(df['bathrooms'], kde=True, ax=axes[0, 1])
        axes[0, 1].set_title('Distribution of Bathrooms')
        sns.histplot(df['sqft_living'], kde=True, ax=axes[0, 2])
        axes[0, 2].set_title('Distribution of Sqft Living')
        sns.histplot(df['grade'], kde=True, ax=axes[1, 0])
        axes[1, 0].set_title('Distribution of Grade')
        sns.histplot(df['yr_built'], kde=True, ax=axes[1, 1])
        axes[1, 1].set_title('Distribution of Year Built')
        plt.tight_layout()
        st.pyplot(f)

        # Correlation heatmap
        st.write("### Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
        st.pyplot(plt)

        # Split data
        X = df.drop(columns='price')
        y = df['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Linear Regression
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)
        y_pred_lr = lin_reg.predict(X_test)

        # Linear Regression Metrics
        st.write("### Linear Regression Results")
        st.write("R² Score:", r2_score(y_test, y_pred_lr))
        st.write("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred_lr))

        # SVR
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        svr = SVR(kernel='rbf', C=100, epsilon=0.1)
        svr.fit(X_train_scaled, y_train)
        y_pred_svr = svr.predict(X_test_scaled)

        # SVR Metrics
        st.write("### Support Vector Regression Results")
        st.write("R² Score:", r2_score(y_test, y_pred_svr))
        st.write("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred_svr))

        # Predict for a new house
        st.write("### Predict House Price")
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)
        sqft_living = st.number_input("Sqft Living", min_value=500, max_value=10000, value=1800)
        grade = st.number_input("Grade", min_value=1, max_value=10, value=7)
        yr_built = st.number_input("Year Built", min_value=1900, max_value=2022, value=1990)

        input_data = [[bedrooms, bathrooms, sqft_living, grade, yr_built]]
        input_scaled = scaler.transform(input_data)

        predicted_price_lr = lin_reg.predict(input_data)
        predicted_price_svr = svr.predict(input_scaled)

        st.write("#### Predicted Price (Linear Regression):", predicted_price_lr[0])
        st.write("#### Predicted Price (SVR):", predicted_price_svr[0])
    else:
        st.error("Dataset must contain columns: 'bedrooms', 'bathrooms', 'sqft_living', 'grade', 'price', and 'yr_built'.")
