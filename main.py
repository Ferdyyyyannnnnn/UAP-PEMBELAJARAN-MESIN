import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import joblib

# Load models and other files
encoder = joblib.load(r'C:\Users\ferdy\OneDrive\Documents\UAPGAES\model\encoder.pkl')
scaler = joblib.load(r'C:\Users\ferdy\OneDrive\Documents\UAPGAES\model\scaler.pkl')
rf_model = joblib.load(r'C:\Users\ferdy\OneDrive\Documents\UAPGAES\model\rf_model.pkl')
xgb_model = joblib.load(r'C:\Users\ferdy\OneDrive\Documents\UAPGAES\model\xgb_model.pkl')
ffnn_model = load_model(r'C:\Users\ferdy\OneDrive\Documents\UAPGAES\model\ffnn_model.h5')

# Load the dataset
file_path = r'C:\Users\ferdy\OneDrive\Documents\UAPGAES\Location.csv'
data = pd.read_csv(file_path)

# Streamlit App Title
st.title('Location Data Model Prediction & Analysis')

# Sidebar for user input
st.sidebar.header('User Input Parameters')

# Select Location
location = st.sidebar.selectbox('Location', data['Location'].unique())

# Select Month
month = st.sidebar.selectbox('Month', data['Month'].unique())

# Select Year
year = st.sidebar.selectbox('Year', data['Year'].unique())

# Select Season
season = st.sidebar.selectbox('Season', data['Season'].unique())

# Safe encoding function
def safe_transform(encoder, label):
    if label in encoder.classes_:
        return encoder.transform([label])[0]
    else:
        return -1  # Default value for unknown labels

# Predict button
if st.sidebar.button('Predict'):
    # Preprocess user input for prediction
    user_data = pd.DataFrame({
        'Location': [location],
        'Season': [season],
        'Month': [month],
        'Year': [year]
    })

    # Encode Location and Season safely
    user_data['Location'] = user_data['Location'].apply(lambda x: safe_transform(encoder, x))
    user_data['Season'] = user_data['Season'].apply(lambda x: safe_transform(encoder, x))

    # Scale the input data
    user_data_scaled = scaler.transform(user_data)

    # Predictions
    rf_pred = rf_model.predict(user_data_scaled)
    xgb_pred = xgb_model.predict(user_data_scaled)
    ffnn_pred = ffnn_model.predict(user_data_scaled)
    ffnn_pred = ffnn_pred.argmax(axis=1)  # Assuming classification with softmax output

    # Display predictions
    st.write(f"Random Forest Prediction: {encoder.inverse_transform(rf_pred)}")
    st.write(f"XGBoost Prediction: {encoder.inverse_transform(xgb_pred)}")
    st.write(f"Feedforward Neural Network Prediction: {encoder.inverse_transform(ffnn_pred)}")

# Show dataset information
if st.checkbox('Show Data Information'):
    st.write(data.head())

# Model performance comparison
st.subheader('Model Performance Comparison')

# Ensure the Location and Season columns are encoded for model accuracy comparison
data['Location'] = data['Location'].apply(lambda x: safe_transform(encoder, x))
data['Season'] = data['Season'].apply(lambda x: safe_transform(encoder, x))

# Prepare data for prediction
X = data[['Location', 'Season', 'Month', 'Year']]
y = data['Season']

# Remove rows with unseen labels (-1)
valid_indices = y != -1
X = X[valid_indices]
y = y[valid_indices]

# Apply scaling to the features
X_scaled = scaler.transform(X)

# Calculate accuracy for Random Forest model
rf_accuracy = accuracy_score(y, rf_model.predict(X_scaled))

# Calculate accuracy for XGBoost model
xgb_accuracy = accuracy_score(y, xgb_model.predict(X_scaled))

# Calculate accuracy for FFNN model
ffnn_pred = ffnn_model.predict(X_scaled)
ffnn_pred = ffnn_pred.argmax(axis=1)  # Convert multioutput to class index
ffnn_accuracy = accuracy_score(y, ffnn_pred)

models = ['Random Forest', 'XGBoost', 'FFNN']
accuracies = [rf_accuracy, xgb_accuracy, ffnn_accuracy]

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(models, accuracies, color=['skyblue', 'orange', 'green'])
ax.set_xlabel('Model')
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Comparison')
ax.set_ylim(0, 1)
st.pyplot(fig)

# Display Visit Count Graphs
if st.checkbox('Show Visit Count Graphs'):
    top_places = data.groupby(['Location', 'Month', 'Year']).size().reset_index(name='Visit_Count')
    top_10_places = top_places.groupby('Location').sum().sort_values(by='Visit_Count', ascending=False).head(10).index
    filtered_data = top_places[top_places['Location'].isin(top_10_places)]

    # Monthly visit count for top 10 places
    st.subheader('Visit Count per Month for Top 10 Places')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.set(style="whitegrid")
    sns.barplot(x="Month", y="Visit_Count", hue="Location", data=filtered_data, ci=None, ax=ax)
    ax.set(title='Top 10 Places: Visit Counts per Month', xlabel='Month', ylabel='Visit Count')
    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    st.pyplot(fig)

    # Yearly visit count for top 10 places
    st.subheader('Visit Count per Year for Top 10 Places')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x="Year", y="Visit_Count", hue="Location", data=filtered_data, ci=None, ax=ax)
    ax.set(title='Top 10 Places: Visit Counts per Year', xlabel='Year', ylabel='Visit Count')
    st.pyplot(fig)

# Display summary of top visited places
if st.checkbox('Show Top Visited Places'):
    top_places_summary = top_places.groupby('Location').agg({'Visit_Count': 'sum'}).reset_index()
    top_places_summary = top_places_summary.sort_values(by='Visit_Count', ascending=False).head(10)
    st.write("Top 10 Most Visited Places:")
    st.write(top_places_summary)
