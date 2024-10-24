# RNN Time Series Forecasting

This project demonstrates the use of a Recurrent Neural Network (RNN) for time series forecasting. The goal is to predict future values based on historical data, showcasing how deep learning can be applied to sequential data. This repository includes data preparation, model training, and evaluation processes.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)

## Overview

Time series forecasting is crucial in various fields, including finance, economics, and meteorology. This project leverages RNNs, particularly Long Short-Term Memory (LSTM) networks, to capture dependencies in sequential data effectively.

## Dataset

The dataset used in this project is a synthetic time series data generated using a sine wave function with added noise. You can easily adapt the code to work with real-world datasets. 

### Data Preparation Steps:
1. **Normalization**: The data is normalized to improve the performance of the neural network.
2. **Windowing**: The dataset is split into sequences (windows) to create input features and corresponding output labels.

## Installation

To run this project, you need to have Python installed along with the following libraries:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

You can also create a virtual environment and install the requirements using:

```bash
# Create a virtual environment (optional)
python -m venv myenv
source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
```

## Usage

1. **Load the Jupyter Notebook**: Open `rnn_time_series_forecasting.ipynb`.
2. **Run the Cells**: Follow the cells in the notebook to:
   - Generate and visualize the synthetic dataset.
   - Prepare the data for training and testing.
   - Train the RNN model on the prepared data.
   - Evaluate the model's performance using RMSE.

3. **Model Saving**: The trained model is saved as `RNN_time_series_forecasting.h5` for future use.

## Model Architecture

The RNN model consists of the following layers:
- **Input Layer**: Accepts the input features.
- **LSTM Layers**: Two LSTM layers with dropout for regularization.
- **Dense Layer**: Outputs the predicted values.

```python
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))
```

## Results

After training, the model's performance is evaluated using RMSE. The results show the effectiveness of the RNN in capturing the patterns within the time series data.

### Example Output:
- Train RMSE: 0.0705

A plot comparing predicted values to actual values is also generated to visualize model performance.

## Conclusion

This project illustrates the implementation of an RNN for time series forecasting. The use of LSTM layers allows the model to learn and predict sequential patterns effectively. 

## Future Work

Future enhancements can include:
- **Hyperparameter Tuning**: Experiment with different hyperparameters to improve model performance.
- **Real-World Data**: Test the model with actual time series datasets.
- **Additional Features**: Explore incorporating exogenous variables or other time series forecasting techniques for improved accuracy.

## Acknowledgements

Thank you for exploring this project! Feel free to contribute, raise issues, or suggest improvements.
