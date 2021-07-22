from sampler import Sampler
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# sample_size=50

def get_data_file_path(company_code):
    return "../company-data/P0717/CompanyData-" + company_code+ "/" + company_code + ".csv"


def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])

    TechIndicator = copy.deepcopy(data)

    TechIndicator['Momentum_1D'] = (TechIndicator['Close'] - TechIndicator['Close'].shift(1)).fillna(0)
    TechIndicator['RSI_14D'] = TechIndicator['Momentum_1D'].rolling(center=False, window=14).apply(rsi).fillna(0)

    TechIndicator['26_ema'] = TechIndicator['Close'].ewm(span=26, min_periods=0, adjust=True, ignore_na=False).mean()
    TechIndicator['12_ema'] = TechIndicator['Close'].ewm(span=12, min_periods=0, adjust=True, ignore_na=False).mean()
    TechIndicator['9_ema'] = TechIndicator['Close'].ewm(span=9, min_periods=0, adjust=True, ignore_na=False).mean()
    TechIndicator['5_ema'] = TechIndicator['Close'].ewm(span=5, min_periods=0, adjust=True, ignore_na=False).mean()
    TechIndicator['2_ema'] = TechIndicator['Close'].ewm(span=2, min_periods=0, adjust=True, ignore_na=False).mean()

    TechIndicator['MACD_2_9'] = (np.tanh((TechIndicator['2_ema'] - TechIndicator['9_ema']) * 1000))
    TechIndicator['MACD_5_12'] = (np.tanh((TechIndicator['5_ema'] - TechIndicator['12_ema']) * 1000))
    TechIndicator['MACD_12_26'] = (np.tanh((TechIndicator['12_ema'] - TechIndicator['26_ema']) * 1000))

    TechIndicator = TechIndicator.fillna(0)

    columns2Drop = [
        'Open', 'Low', 'High',  # 'Close',
        'Momentum_1D',
        '26_ema', '12_ema', '9_ema', '5_ema', '2_ema',  # 'aupband',
        # 'adownband'
    ]
    TechIndicator = TechIndicator.drop(labels=columns2Drop, axis=1)

    data_columns = [
        # 'Open', 'High', 'Low', 'Volume',
        'Close',
        'MACD_2_9',
        'MACD_5_12',
        'MACD_12_26',
        'Volumn',
        'RSI_14D'
    ]

    # X_traning, X_validation, y_training, y_validation = split_data(TechIndicator, data_columns)
    # TechIndicator.insert(0,'-c-',TechIndicator['Close'])
    np_arr =  TechIndicator[data_columns].to_numpy()
    date_arr = TechIndicator['Date'].dt.strftime('%Y-%m-%d').to_numpy()
    return (np_arr, date_arr) 


# def convert_to_arr(data):

#
# def split_data(data, columns):
#     cols = columns.copy()
#     cols.append('Close')
#     input_features = data[cols]
#     input_data = input_features.values
#
#     prices = data['Close'].values
#
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     input_data = scaler.fit_transform(input_data)
#
#     lookback = 14
#     total_size = len(data)
#     X = []
#     y = []
#     for i in range(0, total_size - lookback):  # loop data set with margin 50 as we use 50 days data for prediction
#         t = []
#         for j in range(0, lookback):  # loop for 50 days
#             current_index = i + j
#             t.append(input_data[current_index, :])  # get data margin from 50 days with marging i
#         X.append(t)
#         y.append(prices[lookback + i])
#
#     X, y = np.array(X), np.array(y)
#
#     X_traning, X_validation, y_training, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     X_traning = X_traning.reshape(X_traning.shape[0], lookback, len(cols))
#     X_validation = X_validation.reshape(X_validation.shape[0], lookback, len(cols))
#
#     return X_traning, X_validation, y_training, y_validation


def rsi(values):
    up = values[values > 0].mean()
    down = -1 * values[values < 0].mean()
    return 100 * up / (up + down)


class CompanyDataSampler(Sampler):

    def __init__(self, company_code="AEL",window_episode=None):
        self.n_var = 6
        self.company_code = company_code
        self.sample = self.__load_my_data
        self.title = "CompanyData-" + company_code
        self.window_episode=window_episode

    def __load_my_data(self):
        file_path = get_data_file_path(self.company_code)
        return load_data(file_path), self.title
