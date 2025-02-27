import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# load data
data = pd.read_csv(r"C:\Users\Jan\Downloads\HistoricalQuotes.csv")
data["Close/Last"] = data["Close/Last"].str.replace("$", "").astype(float)
close_prices = data["Close/Last"].values

# split into train/test
train_data, test_data = train_test_split(close_prices, test_size=0.2, shuffle=False)


def calculate_mse(true_values, predicted_values):
    min_length = min(len(true_values), len(predicted_values))
    true_values = true_values[:min_length]
    predicted_values = predicted_values[:min_length]
    return mean_squared_error(true_values, predicted_values)


def second_order_fir_filter(data, order, bias=False):
    # Irandomized coefficients
    w = np.random.rand(order)
    if bias:
        b = np.random.rand(1)

    x = np.zeros((len(data) - order, order))
    for i in range(order):
        x[:, i] = data[i : i + len(data) - order]

    # filter
    predicted_values = np.zeros(len(data) - order)
    for i in range(len(data) - order):
        if bias:
            predicted_values[i] = np.dot(w, x[i]) + b
        else:
            predicted_values[i] = np.dot(w, x[i])

        # lms update
        e = data[i + order] - predicted_values[i]
        w += 0.01 * e * x[i]
        if bias:
            b += 0.01 * e

    return predicted_values


# a)
predicted_values_a = second_order_fir_filter(train_data, order=2)
mse_a = calculate_mse(test_data[2:], predicted_values_a)

# b)
predicted_values_b = second_order_fir_filter(train_data, order=2, bias=True)
mse_b = calculate_mse(test_data[2:], predicted_values_b)

# c)
predicted_values_c = second_order_fir_filter(train_data, order=3)
mse_c = calculate_mse(test_data[3:], predicted_values_c)

# d)
predicted_values_d = second_order_fir_filter(train_data, order=3, bias=True)
mse_d = calculate_mse(test_data[3:], predicted_values_d)

print("MSE for 2nd order adaptive FIR filter without bias:", mse_a)
print("MSE for 2nd order adaptive FIR filter with bias:", mse_b)
print("MSE for 3rd order adaptive FIR filter without bias:", mse_c)
print("MSE for 3rd order adaptive FIR filter with bias:", mse_d)

# e)
min_mse = min(mse_a, mse_b, mse_c, mse_d)
if min_mse == mse_a:
    print("The 2nd order adaptive FIR filter without bias is the best predictor.")
elif min_mse == mse_b:
    print("The 2nd order adaptive FIR filter with bias is the best predictor.")
elif min_mse == mse_c:
    print("The 3rd order adaptive FIR filter without bias is the best predictor.")
else:
    print("The 3rd order adaptive FIR filter with bias is the best predictor.")
