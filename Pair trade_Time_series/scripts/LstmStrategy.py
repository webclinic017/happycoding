import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import keras


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
    #print(np.abs(np.max(data) - np.min(data)))
    if np.abs(np.max(data) - np.min(data)) < 0.15:
        return '-'
    judge = len(data)  # depends on how many data points we are looking up
    diff1 = differentialHelper(data)
    ndiff1 = np.array(diff1)

    if np.sum(ndiff1[:3]) > 2:
        return 'UP'
    if np.sum(ndiff1[:3]) < -2:
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
    transaction_list = [[-1, '-', '-', 0, '-']]  # initial first line of transaction

    for row in trend_prediction:
        profit = 0
        current_trend = str(row[1])
        lastest_position = transaction_list[-1]
        prev_strategy = lastest_position[2]
        action, buyOrShort = positionStrategy(current_trend, lastest_position[4])
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
            transaction_list.append([row[0], 'C', prev_strategy, profit, current_trend])
        if action.endswith('O'):
            # if previous one has not closed yet, should be error!!!
            if (transaction_list[-1][1] == 'O'):
                print("ERROR!!!!! CHECK POINT 1")
            transaction_list.append([row[0], 'O', buyOrShort, 0, current_trend])
    return profit_cumulative, transaction_list


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
    model.add(Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.RandomNormal(seed=sd),
                    bias_initializer=keras.initializers.Constant(value=0.1)))
    return model
