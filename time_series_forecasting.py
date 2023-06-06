import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

class TimeSeriesForecaster:
    def __init__(self, data):
        self.data = data

    def fit_arima(self, order):
        model = ARIMA(self.data, order=order)
        self.arima_model = model.fit()

    def forecast(self, num_steps):
        forecast_values = self.arima_model.forecast(steps=num_steps)
        return forecast_values

if __name__ == '__main__':
    # Example time series data
    data = pd.Series([12, 15, 18, 20, 22, 25, 28, 30])

    forecaster = TimeSeriesForecaster(data)

    order = (1, 0, 0)  # ARIMA order (p, d, q)
    forecaster.fit_arima(order)

    num_steps = 3  # Number of steps to forecast
    forecast_values = forecaster.forecast(num_steps)

    print("Forecasted values:")
    print(forecast_values)
