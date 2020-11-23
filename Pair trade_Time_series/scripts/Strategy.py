import numpy as np


# model: trained model
# data:testing data, real data
# start:from which point to predict
def getPredictionAndPrevRealData(model, data, start, window_size=3):
    real, pred = [], []
    real.append(data.iloc[start - window_size:start + 1, 4])
    for_forecast = model.get_prediction(start=start, end=start + window_size, dynamic=True)
    for_forecast = for_forecast.predicted_mean
    pred.append(for_forecast[:])
    return real, pred

def getRealData(data, start, window_size=3):
    real=[]
    real.append(data.iloc[start:start + window_size+1, 4])
    return real

# if forcast is going up and the 1st differential is also goind up, it's an uptrend,
# and if previous is also uptrend, then this trend continues

# previous trend: UP -> uptrend; DOWN-> downtrend
def trendPredictor(previousTrend, real, forcast):
    nextTrend = trendHelper(forcast)
    if previousTrend == nextTrend:
        return True, previousTrend
    else:
        return False, nextTrend


# first return is the 1 st differential, how many are positive;
# 2nd return is the 2nd differential, how many are positive
def trendHelper(data):
    data = np.array(data).ravel()
    judge = len(data)  # depends on how many data points we are looking up
    diff1 = differentialHelper(data)
    ndiff1 = np.array(diff1)
    #print(f'1:{ndiff1}')
    diff2 = differentialHelper(diff1)
    ndiff2 = np.array(diff2)
    #print(f'2:{ndiff2}')
    if (np.sum(ndiff1 > 0) > judge - 2):
        return 'UP'
    elif (np.sum(ndiff1 > 0) > judge - 3 and np.sum(ndiff2 > 0) > judge - 4):
        return 'UP'
    elif (np.sum(ndiff1 < 0) > judge - 2):
        return 'DOWN'
    elif (np.sum(ndiff1 < 0) > (judge - 3) and np.sum(ndiff2 < 0) > (judge - 4)):
        return 'DOWN'
    else:
        return '-'


def differentialHelper(data):
    threshold = 0.5 #less than 0.5% change will be ignored
    diff = []
    for i in range(1, len(data)):
        if data[i - 1] == 0:
            diff.append(0)
        else:
            change = ((data[i] - data[i - 1]) / data[i - 1])
            if np.abs(change * 100) < threshold:
                change = 0
            diff.append(change)
    return diff


# return next position based on our prediction, right?
# first return is O/C/- for open close or not action needed
# second return L/S for long or short
def positionStrategy(currentPosition, inPositionTrend, real, prediction):
    currentTrend = trendHelper(real)
    trendContinue, nextTrend = trendPredictor(currentTrend, real, prediction)
    # print(f'trend continue?{trendContinue}, currentTrend {currentTrend} nextTrend {nextTrend}')
    action = ''
    buyOrShort = ''
    if (nextTrend == 'UP'):
        buyOrShort = 'L'
    elif (nextTrend == 'DOWN'):
        buyOrShort = 'S'
    else:
        buyOrShort = '-'
    if currentPosition == '-' and nextTrend == 'UP':
        return 'O', 'L'
    if currentPosition == '-' and nextTrend == 'DOWN':
        return 'O', 'S'
    if inPositionTrend == nextTrend:  # trend will continue, don't have to do anything
        return '-', ''
    if nextTrend == '-':  # what should I do if next trend is not predictable? now doing nothing
        return '-', ''

    else:  # current position is O/C
        if currentPosition == 'O':
            action = 'C'
        else:
            action = 'O'

    return action, buyOrShort


def calc_ols(df1, s1, s2, smoothing):
    pair = df1.iloc[:, [s1, s2]]
    pair.dropna(how='any', axis=0, inplace=True)

    if smoothing == 'yes':
        pair = pair.rolling(window=5).mean()
        pair.dropna(inplace=True)

    c = np.polyfit(pair.iloc[:, 0], pair.iloc[:, 1], 1, full=False)
    temp_list = pair.iloc[:, 1] - c[0] * pair.iloc[:, 0]

    return c[0], temp_list


def up_gap(test_data, cp, coef):
    high_position = test_data[cp + 3, 2]
    low_position = test_data[cp + 3, 1]

    return position


def down_gap(test_data, cp, coef):
    position = coef * test_data[cp + 3, 1] - test_data[cp + 3, 2]

    return position
