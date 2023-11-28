# This code demonstrates the process of chaining AI models for network traffic prediction and anomaly detection. 
# The LSTM model is used to forecast future traffic patterns, while the Isolation Forest model identifies anomalous traffic deviations.
# The predicted traffic and detected anomalies are then analyzed to trigger appropriate actions, such as resource allocation or anomaly investigation.
# Author: Rani Yadav-Ranjan
# Nov. 28, 2023

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from sklearn.ensemble import IsolationForest

# Load network traffic data from a CSV file
data = pd.read_csv('network_traffic_data.csv')

# Preprocess data to ensure compatibility with AI models
data_preprocessed = MinMaxScaler().fit_transform(data)

# Split data into training and testing sets
train_data = data_preprocessed[:int(0.8 * len(data_preprocessed))]
test_data = data_preprocessed[int(0.8 * len(data_preprocessed)):]

# Train the LSTM time series forecasting model
model_lstm = keras.Sequential([
    keras.layers.LSTM(64, return_sequences=True, input_shape=(data_preprocessed.shape[1], 1)),
    keras.layers.LSTM(32),
    keras.layers.Dense(1)
])
model_lstm.compile(loss='mean_squared_error', optimizer='adam')
model_lstm.fit(train_data, data['traffic_volume'][:int(0.8 * len(data))], epochs=10)

# Train the Isolation Forest anomaly detection model
model_isolation_forest = IsolationForest()
model_isolation_forest.fit(train_data)

# Make predictions on the test data
predicted_traffic = model_lstm.predict(test_data)

# Evaluate the performance of the LSTM model
mean_squared_error = model_lstm.evaluate(test_data, data['traffic_volume'][int(0.8 * len(data)):])
print('Mean Squared Error:', mean_squared_error)

# Detect anomalies using the Isolation Forest model
anomaly_scores = model_isolation_forest.predict_score(test_data)

# Identify anomalous data points
anomalies = [anomaly_score < 0 for anomaly_score in anomaly_scores]

# Analyze and respond to predicted traffic and detected anomalies
for i in range(len(anomalies)):
    if predicted_traffic[i] > threshold:
        # Allocate additional network resources to handle the predicted increase in traffic
        allocate_resources()

    if anomalies[i]:
        # Investigate the detected anomaly to identify its root cause and take appropriate action
        investigate_anomaly()
