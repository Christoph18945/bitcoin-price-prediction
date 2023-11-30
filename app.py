#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Predict Bitcoin price"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
import matplotlib.pyplot as plt
import matplotlib.pyplot as t
import matplotlib.dates as mdates

def main() -> None:
    """main funtion"""
    prepare_data()
    predict_bitcoin_course()

def prepare_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """prepare data for neuronal network (LSTM)"""
    # read csv file in and mark first row as header
    pandas_data_frame: pd.DataFrame = pd.read_csv(r"data/Kraken_BTCUSD_d.csv", header = 0)
    
    # select the "Open" column from the csv file, move all lines in the
    # column upwardsmove the result to a new column named "Open_before".
    # the shift is done because the value in Open column must be compared
    # with its preceeding value now on the same line in the Open_before column.
    pandas_data_frame["Open_before"] = pandas_data_frame["Open"].shift(-1)
    
    print(pandas_data_frame.head())
    #          Date  Symbol    Open    High     Low   Close  Volume BTC  Volume USD  Open_before
    # 0  2013-10-24  BTCUSD  202.24  207.30  169.92  185.95       83.74    16514.45       185.95
    # 1  2013-10-25  BTCUSD  185.95  193.10  171.97  183.00       67.20    12182.36       183.00
    # 2  2013-10-26  BTCUSD  183.00  186.05  179.87  184.58       12.00     2211.39       184.58
    # 3  2013-10-27  BTCUSD  184.58  191.30  179.24  191.00       69.91    12910.72       191.00
    
    # here it is calculated whether the price has risen or fallen.
    # we dived the value from the Open_before column through the value
    # in the Open column and subtract 1. It means:
    # (202.24 / 185.95) - 1 = 0.0876... means the price decreased by 0.0876 %
    pandas_data_frame["Open_after"] = (pandas_data_frame["Open"] / pandas_data_frame["Open_before"]) - 1

    # save the all calculated values in the changes variable
    changes = pandas_data_frame["Open_after"]

    # input shape ()nuber_of_training_example, sequence_length, input_data)
    X: list = [] # added values in reverse order
    Y: list = [] # added values

    for i in range(0, len(changes) - 20):
        Y.append(changes[i])
        # changes[i+1:i+21] retrieves the first 20 values
        # [::-1] reverses the whole list
        X.append(np.array(changes[i+1:i+21][::-1]))

    # convert X and Y in a numpy array
    X = np.array(X).reshape(-1, 20, 1)
    Y = np.array(Y)

    return X, Y, pandas_data_frame

def predict_bitcoin_course() -> None:
    """train LSTM to predict the bitcoin course"""
    # create LSTM model
    input_data = prepare_data()

    # instantiate a keras sequential model
    model: Sequential = Sequential()
    
    # add a layer to it. 1 LSTM cell (neuron)
    # (20, 1) means the model expects an input sequence
    # of 20 with one feature.
    model.add(LSTM(1, input_shape=(20, 1)))

    # Configure the model for the training: 
    # optimizer="rmsprop" specifies the optimization
    # algorithm. Here th ealgorithm is called Root Mean
    # Square Propagation.
    # loss="mse" specificies the loss function. "mse" stands
    # for Mean Square Error (heavily used for regression problems).
    model.compile(optimizer="rmsprop", loss="mse")

    # The actual training with the input data:
    # input_data[0] is the features that the model learn the patterns from
    # input_data[1] is the labels corresponding the input data
    # batch_size=32 means the number of samples used in each iteration
    # (batch) for updating model weights
    # epochs=10 means 10 times the model will iterate overthe entire
    # dataset during the training
    model.fit(input_data[0], input_data[1], batch_size=32, epochs=10)

    # predict value based on the 20 values provided above
    predictions = model.predict(input_data[0])
    
    # reshaping the predictions array. The -1 argument in
    # the reshape method is used to automatically infer the
    # size of one of the dimensions, based on the size of the
    # original array.
    predictions = predictions.reshape(-1)

    # 
    predictions = np.append(predictions, np.zeros(20))

    # 
    input_data[2]["predictions"] = predictions
    input_data[2]["Open_predicted"] = input_data[2]["Open_before"] * (1 + input_data[2]["predictions"])

    # 
    years = mdates.YearLocator()
    months = mdates.MonthLocator()

    # set figure size
    plt.figure(figsize=(10, 6))

    # rotate labels on x-axis to make it
    # align it properly and enhance readability
    plt.xticks(rotation=90)

    # Use AutoDateLocator to automatically format date labels.
    ax = plt.gca()
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_locator(months)

    # 
    plt.plot(input_data[2]["Date"][-1000:], input_data[2]["Open"][-1000:], label="Open")
    plt.plot(input_data[2]["Date"][-1000:], input_data[2]["Open_predicted"][-1000:], label="Open (predicted)")

    # add a legend to the plot with
    # a small fontsize
    plt.legend(fontsize='small')

    # save result plot
    plt.savefig(r'img/result.png')

    plt.show()

    return None

if __name__ == "__main__":
    main()
