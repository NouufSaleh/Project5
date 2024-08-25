import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Load the GRU model without compiling
model = load_model('gru_model.h5', compile=False)

# Load your dataset, ensuring it's indexed correctly
df = pd.read_csv('traffic.csv', parse_dates=['DateTime'])
df.drop_duplicates(subset='DateTime', keep='last', inplace=True)
df.set_index('DateTime', inplace=True)
df = df.resample('H').mean()  # Resample to ensure hourly intervals
df.interpolate(method='time', inplace=True)  # Interpolate missing values

# Initialize and fit the MinMaxScaler using all available data
scaler = MinMaxScaler()
scaler.fit(df[['Vehicles']])  # Adjust the column name as necessary

# Streamlit app interface
st.title('Vehicle Count')
selected_date = st.date_input("Choose a date for prediction", datetime.now())
selected_date = pd.Timestamp(selected_date)  # Ensuring it's a Timestamp object

if st.button('Predict'):
    predictions = []
    last_known_data = df[-72:]['Vehicles'].to_numpy()  # Last 72 hours of known data
    for hour in range(24):
        current_hour = datetime(selected_date.year, selected_date.month, selected_date.day, hour)
        
        if current_hour <= df.index.max():
            # Use actual historical data when available
            start_date = current_hour - timedelta(hours=72)
            end_date = current_hour - timedelta(seconds=1)

            input_sequence = df.loc[start_date:end_date]['Vehicles'].to_numpy()
        else:
            # Use the last known data and recursive predictions
            input_sequence = last_known_data
            last_prediction = model.predict(scaler.transform(input_sequence.reshape(-1, 1)).reshape(1, 72, 1))
            last_prediction = scaler.inverse_transform(last_prediction).flatten()[0]
            input_sequence = np.append(input_sequence[1:], last_prediction)
            last_known_data = input_sequence  # Update the rolling window with the new prediction

        if len(input_sequence) != 72:
            st.error(f"Expected 72 data points but got {len(input_sequence)}.")
            break

        # Scaling and reshaping the data for the model
        input_sequence_scaled = scaler.transform(input_sequence.reshape(-1, 1))
        input_sequence_scaled = input_sequence_scaled.reshape(1, 72, 1)

        # Prediction
        prediction = model.predict(input_sequence_scaled)
        prediction_inverse = scaler.inverse_transform(prediction)

        # Storing predictions
        predictions.append((current_hour, prediction_inverse.flatten()[0]))

    # Displaying predictions
    if predictions:
        for hour, pred in predictions:
            st.write(f"Predicted vehicle count for {hour.strftime('%Y-%m-%d %H:%M:%S')}: {pred:.0f}")

