import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def ticket_forecast(file_name="clean-data.csv"):

    data = pd.read_csv(file_name)
    data["Date"] = pd.to_datetime(data["Date"])
    data.set_index("Date", inplace=True)
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    model_fit = model.fit()
    forecast = model_fit.get_forecast(steps=28)
    forecast_index = pd.date_range(
        data.index[-1] + pd.DateOffset(days=1), periods=28, freq="D"
    )
    forecast_df = pd.DataFrame(
        {"n_tickets": forecast.predicted_mean.values}, index=forecast_index
    )
    forecast_df["n_tickets"] = forecast_df["n_tickets"].round().astype(int)
    forecast_df.to_csv("forecast-data.csv")


if __name__ == "__main__":
    ticket_forecast(file_name="clean-data.csv")
