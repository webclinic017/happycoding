import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, RNN
import keras
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def mape_score(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def shortProfitCalculator(test_data, open_idx, close_idx):
    a = test_data.iloc[open_idx, 5]
    b = test_data.iloc[close_idx, 5]
    return a - b


def longProfitCalculator(test_data, open_idx, close_idx):
    a = test_data.iloc[open_idx, 5]
    b = test_data.iloc[close_idx, 5]
    return b - a


# first return is the 1 st differential, how many are positive;
# 2nd return is the 2nd differential, how many are positive
def trendHelper(data):
    data = np.array(data).ravel()
    # make this rule here: if abs change not much in this data, then predict no trend
    # because after normalize, small change on smaller number is huge
    # print(np.abs(np.max(data) - np.min(data)))
    if np.abs(np.max(data) - np.min(data)) < 0.15:
        return '-'
    judge = len(data)  # depends on how many data points we are looking up
    diff1 = differentialHelper(data)
    ndiff1 = np.array(diff1)
    non_zero_index = -1
    for non_zero_index, x in enumerate(ndiff1):
        if x != 0:
            break
    if np.sum(ndiff1[non_zero_index:non_zero_index + 3]) > 2:
        return 'UP'
    if np.sum(ndiff1[non_zero_index:non_zero_index + 3]) < -2:
        return 'DOWN'
    # print(f'1:{ndiff1}')

    num_gt_0 = np.sum(ndiff1 > 0)
    num_lt_0 = np.sum(ndiff1 < 0)
    gt0sum = 0
    lt0sum = 0
    for x in diff1:
        if x > 0:
            gt0sum += x
        if x < 0:
            lt0sum += x

    if num_gt_0 > num_lt_0:
        num_gt_0 = num_gt_0 - num_lt_0
        if num_gt_0 >= 3 and np.sum(diff1) > 0:
            return 'UP'
        if lt0sum != 0 and np.abs(gt0sum / lt0sum) > 2.5:
            return 'UP'
        else:
            return '-'
    elif num_gt_0 < num_lt_0:
        num_lt_0 = num_lt_0 - num_gt_0
        if num_lt_0 >= 3 and np.sum(diff1) < 0:
            return 'DOWN'
        if gt0sum != 0 and np.abs(lt0sum / gt0sum) > 2.5:
            return 'DOWN'
        else:
            return '-'
    else:
        return '-'


def differentialHelper(data):
    threshold = 0.25  # less than 25% change will be ignored, I just want big sharp prediction to ge captured
    diff = []
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / std
    for i in range(1, len(data)):
        t = data[i] - data[i - 1]
        if np.abs(t) > threshold:
            diff.append(t)
        else:
            diff.append(0)

    # if data[i - 1] == 0:
    #   diff.append(0)
    # else:
    #    change = ((data[i] - data[i - 1]) / data[i - 1])
    #    if np.abs(change * 100) < threshold:
    #        change = 0
    #    diff.append(change)
    return diff


# return next position based on our prediction, right?
# first return is O/C/- for open close or not action needed
# second return L/S for long or short
def positionStrategy(currentTrend, inPositionTrend):
    currentTrend = str(currentTrend)
    inPositionTrend = str(inPositionTrend)
    # print(f'current: {currentTrend} in position trend {inPositionTrend}')
    if currentTrend.startswith('UP'):
        currentTrend = "UP"
    if currentTrend.startswith('DOWN'):
        currentTrend = "DOWN"
    if inPositionTrend.startswith('UP'):
        inPositionTrend = "UP"
    if inPositionTrend.startswith('DOWN'):
        inPositionTrend = "DOWN"
    action = ''
    buyOrShort = ''

    if (currentTrend == 'UP'):
        buyOrShort = 'L'
    elif (currentTrend == 'DOWN'):
        buyOrShort = 'S'
    else:
        buyOrShort = '-'
    if currentTrend == inPositionTrend:
        return '-', '-'
    if currentTrend == '-' and inPositionTrend != '-':
        return '-', buyOrShort
    if currentTrend != '-' and inPositionTrend == '-':
        return 'O', buyOrShort
    if currentTrend == 'UP':
        return 'CO', buyOrShort
    if currentTrend == 'DOWN':  # what should I do if next trend is not predictable? now doing nothing
        return 'CO', buyOrShort

    return action, buyOrShort


def trend_prediction_to_transaction(trend_prediction, test_data):
    profit_cumulative = 0
    transaction_list = [[-1, '-', '-', '-', 0, '-', 0]]  # initial first line of transaction

    for row in trend_prediction:
        date = test_data.iloc[row[0], 0]
        profit = 0
        current_trend = str(row[2])
        lastest_position = transaction_list[-1]
        prev_strategy = lastest_position[3]
        action, buyOrShort = positionStrategy(current_trend, lastest_position[5])
        action = str(action)
        # print(current_trend)
        if action.startswith('C'):
            # FIRST NEED TO LOSE AND THEN ENTER NEW POSITION
            open_idx = lastest_position[0]
            if prev_strategy == 'L':
                profit = longProfitCalculator(test_data, open_idx, row[0])
            if prev_strategy == 'S':
                profit = shortProfitCalculator(test_data, open_idx, row[0])
            # print(f'closed. idx {open_idx} to idx {row[0]}. profit {profit}')
            profit_cumulative = profit_cumulative + profit
            transaction_list.append(
                [row[0], date, 'C', prev_strategy, profit, current_trend, test_data.iloc[row[0], -1]])
        if action.endswith('O'):
            # if previous one has not closed yet, should be error!!!
            if (transaction_list[-1][1] == 'O'):
                print("ERROR!!!!! CHECK POINT 1")
            transaction_list.append([row[0], date, 'O', buyOrShort, 0, current_trend, test_data.iloc[row[0], -1]])
    return profit_cumulative, transaction_list

#withe 200 days this model is best for TW pair
def build_model(look_back, sd):
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, look_back), activation='relu', return_sequences=True,
                   kernel_initializer=keras.initializers.RandomNormal(seed=sd),
                   bias_initializer=keras.initializers.Constant(value=0.1)))
    # model.add(LSTM(128,activation='tanh',return_sequences=True))
    model.add(LSTM(16, activation='tanh',
                   kernel_initializer=keras.initializers.RandomNormal(seed=sd),
                   bias_initializer=keras.initializers.Constant(value=0.1)))
    # model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu', kernel_initializer=keras.initializers.RandomNormal(seed=sd),
                    bias_initializer=keras.initializers.Constant(value=0.1)))
    model.add(Dropout(0.1, seed=sd))
    model.add(Dense(1, kernel_initializer=keras.initializers.RandomNormal(seed=sd),
                    bias_initializer=keras.initializers.Constant(value=0.1)))
    return model


def build_model_v2(look_back, sd):
    model_inputs = keras.Input(shape=(1, look_back))

    model_lstm_layer = LSTM(16, input_shape=(1, look_back), activation='relu', return_sequences=False,
                            kernel_initializer=keras.initializers.RandomNormal(seed=sd),
                            bias_initializer=keras.initializers.Constant(value=0.1))(model_inputs)
    # model_lstm_layer = LSTM(16, activation='tanh',
    #                         kernel_initializer=keras.initializers.RandomNormal(seed=sd),
    #                         bias_initializer=keras.initializers.Constant(value=0.1))(model_lstm_layer)
    model_lstm_layer = Dropout(0.1, seed=sd)(model_lstm_layer)

    model_nn_layer = tf.keras.layers.Flatten()(model_inputs)
    model_nn_layer = Dense(4, activation='sigmoid', kernel_initializer=keras.initializers.RandomNormal(seed=sd),
                           bias_initializer=keras.initializers.Constant(value=0.1))(model_nn_layer)

    merged = keras.layers.concatenate([model_lstm_layer, model_nn_layer], axis=1)
    merged = Dense(1, kernel_initializer=keras.initializers.RandomNormal(seed=sd),
                   bias_initializer=keras.initializers.Constant(value=0.1))(merged)
    model = keras.Model(inputs=[model_inputs], outputs=merged)
    return model


def build_model_v2_backup(look_back, sd):
    model_inputs = keras.Input(shape=(1, look_back))

    model_lstm_layer = LSTM(32, input_shape=(1, look_back), activation='relu', return_sequences=False,
                            kernel_initializer=keras.initializers.RandomNormal(seed=sd),
                            bias_initializer=keras.initializers.Constant(value=0.1))(model_inputs)
    # model_lstm_layer = LSTM(16, activation='tanh',
    #                         kernel_initializer=keras.initializers.RandomNormal(seed=sd),
    #                         bias_initializer=keras.initializers.Constant(value=0.1))(model_lstm_layer)
    model_lstm_layer = Dropout(0.1, seed=sd)(model_lstm_layer)

    model_nn_layer = tf.keras.layers.Flatten()(model_inputs)
    model_nn_layer = Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.RandomNormal(seed=sd),
                           bias_initializer=keras.initializers.Constant(value=0.1))(model_nn_layer)

    merged = keras.layers.concatenate([model_lstm_layer, model_nn_layer], axis=1)
    merged = Dense(1, kernel_initializer=keras.initializers.RandomNormal(seed=sd),
                   bias_initializer=keras.initializers.Constant(value=0.1))(merged)
    model = keras.Model(inputs=[model_inputs], outputs=merged)
    return model


def mape_score(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def model_selection(test_data, START=0, END=100, look_back=7, SPLIT=0.2, sd=123, future_size=10,
                    select_col='realcombo'):
    dataset = test_data.loc[START:END, select_col].to_numpy()
    dataset = np.nan_to_num(dataset)

    dataset = np.reshape(dataset, (dataset.shape[0], 1, 1))
    print(dataset.shape)
    x, y = create_dataset(dataset[:], look_back)
    x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    y = np.reshape(y, (y.shape[0], y.shape[1]))

    size = len(x)
    train_size = int(size * (1 - SPLIT))
    test_size = size - train_size
    x_train, x_test = x[0:train_size, :, :], x[train_size:, :, :]
    y_train, y_test = y[0:train_size], y[train_size:]
    print(f"train size LSTM {x_train.shape}")
    print(f"test size LSTM {x_test.shape}")

    model = build_model_v2(look_back, sd)
    # model_inputs = keras.Input(shape=(1, look_back))
    # model_lstm_layer = LSTM(64, input_shape=(1, look_back), activation='relu', return_sequences=True,
    #                         kernel_initializer=keras.initializers.RandomNormal(seed=sd),
    #                         bias_initializer=keras.initializers.Constant(value=0.1))(model_inputs)
    # model_lstm_layer = LSTM(16, activation='tanh',
    #                         kernel_initializer=keras.initializers.RandomNormal(seed=sd),
    #                         bias_initializer=keras.initializers.Constant(value=0.1))(model_lstm_layer)
    # model_lstm_layer = Dropout(0.1, seed=sd)(model_lstm_layer)
    #
    # model_nn_layer = tf.keras.layers.Flatten()(model_inputs)
    # model_nn_layer = Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.RandomNormal(seed=sd),
    #                        bias_initializer=keras.initializers.Constant(value=0.1))(model_nn_layer)
    #
    # merged = keras.layers.concatenate([model_lstm_layer, model_nn_layer], axis=1)
    # merged = Dense(1,  kernel_initializer=keras.initializers.RandomNormal(seed=sd),
    #                bias_initializer=keras.initializers.Constant(value=0.1))(merged)
    # model = keras.Model(inputs=[model_inputs], outputs=merged)
    # model.summary()
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), batch_size=1, callbacks=[es], verbose=1)

    y_predict = model.predict(x_test)
    score_test = mape_score(y_test, y_predict)
    print(f'score on test data is: {score_test:.2f}%')

    train_predict = model.predict(x_train)
    score_train = mape_score(y_train, train_predict)
    print(f'score on all train data is: {score_train:.2f}%')

    future = []
    latest_data = dataset.ravel()
    latest_data = latest_data[-look_back:]
    next_pred = model.predict(latest_data.reshape(-1, 1, look_back))
    future.append(next_pred[0][0])
    while len(future) < future_size:
        latest_data = latest_data[-look_back + 1:]
        latest_data = np.append(latest_data, next_pred)
        next_pred = model.predict(latest_data.reshape(-1, 1, look_back))
        future.append(next_pred[0][0])

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset[:, 0, 0])
    trainPredictPlot[:] = np.nan
    trainPredictPlot[look_back:len(train_predict) + look_back] = train_predict[:, 0]
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset[:, 0, 0])
    testPredictPlot[:] = np.nan
    testPredictPlot[len(train_predict) + (look_back):len(dataset[:, 0, 0])] = y_predict[:, 0]

    # shift future predictions for plotting
    futurePredictPlot = np.zeros((dataset.shape[0] + future_size,))
    futurePredictPlot[:] = np.nan
    futurePredictPlot[dataset.shape[0]:dataset.shape[0] + future_size] = future[:]

    # plot baseline and predictions
    true_data, = plt.plot(test_data.loc[START:END + future_size, select_col].to_numpy(), color='g', label='truth data')
    train_data, = plt.plot(trainPredictPlot, color='b', marker='o', label='train data')
    predict_data, = plt.plot(testPredictPlot, color='r', marker='o', linestyle='dashed', label='predict data')
    temp, = plt.plot(futurePredictPlot, color='k', marker='o', linestyle='dashed', label='? data')
    plt.legend(handles=[true_data, train_data, predict_data, temp])
    plt.title(select_col)
    plt.grid()
    plt.show()
    return future


def build_model_and_predict_and_show(test_data, START, END, look_back=7, future_size=10, select_col='realcombo',
                                     seed=123):
    dataset = test_data.loc[START:END, select_col].to_numpy()
    dataset = np.nan_to_num(dataset)
    # print(dataset.shape)  ##NOTE: END IS INCLUDED
    dataset = np.reshape(dataset, (dataset.shape[0], 1, 1))
    x, y = create_dataset(dataset[:], look_back)
    x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    y = np.reshape(y, (y.shape[0], y.shape[1]))

    size = len(x)

    model = build_model_v2(look_back, seed)
    # model.summary()
    # simple early stopping
    es = EarlyStopping(monitor='loss', patience=5, verbose=1)
    model.compile(optimizer='adam', loss='mse', metrics=['acc'])
    model.fit(x, y, epochs=150, batch_size=1, callbacks=[es], verbose=0)

    train_predict = model.predict(x)
    score_train = mape_score(y, train_predict)
    print(f'score on all train data is: {score_train:.2f}%')

    future = []
    latest_data = dataset.ravel()
    latest_data = latest_data[-look_back:]
    next_pred = model.predict(latest_data.reshape(-1, 1, look_back))
    future.append(next_pred[0][0])
    while len(future) < future_size:
        latest_data = latest_data[-look_back + 1:]
        latest_data = np.append(latest_data, next_pred)
        next_pred = model.predict(latest_data.reshape(-1, 1, look_back))
        future.append(next_pred[0][0])

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset[:, 0, 0])
    trainPredictPlot[:] = np.nan
    trainPredictPlot[look_back:len(train_predict) + look_back] = train_predict[:, 0]
    # shift future predictions for plotting
    futurePredictPlot = np.zeros((dataset.shape[0] + future_size,))
    futurePredictPlot[:] = np.nan
    futurePredictPlot[dataset.shape[0]:dataset.shape[0] + future_size] = future[:]

    # plot baseline and predictions
    true_data, = plt.plot(test_data.loc[START:END + future_size, select_col].to_numpy(), color='g', label='truth data')
    train_data, = plt.plot(trainPredictPlot, color='b', marker='o', label='train data')
    future_data, = plt.plot(futurePredictPlot, color='k', marker='o', linestyle='dashed', label='? data')
    plt.legend(handles=[true_data, train_data, future_data])
    plt.title(select_col)
    plt.grid()
    plt.show()
    return future


# predict future
def build_model_and_predict(test_data, START, END, look_back=7, future_size=10, select_col='realcombo', seed=123):
    dataset = test_data.loc[START:END, select_col].to_numpy()
    dataset = np.nan_to_num(dataset)
    dataset = np.reshape(dataset, (dataset.shape[0], 1, 1))
    x, y = create_dataset(dataset[:], look_back)
    x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
    y = np.reshape(y, (y.shape[0], y.shape[1]))

    size = len(x)

    model = build_model_v2(look_back, seed)
    # model.summary()
    # simple early stopping
    es = EarlyStopping(monitor='loss', patience=5, verbose=0)
    model.compile(optimizer='adam', loss='mse', metrics=['acc'])
    model.fit(x, y, epochs=150, batch_size=1, callbacks=[es], verbose=0)

    future = []
    latest_data = dataset.ravel()
    latest_data = latest_data[-look_back:]
    next_pred = model.predict(latest_data.reshape(-1, 1, look_back))
    future.append(next_pred[0][0])
    while len(future) < future_size:
        latest_data = latest_data[-look_back + 1:]
        latest_data = np.append(latest_data, next_pred)
        next_pred = model.predict(latest_data.reshape(-1, 1, look_back))
        future.append(next_pred[0][0])

    return future
