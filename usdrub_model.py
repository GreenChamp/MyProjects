import numpy as np
import pandas as pd
!pip3 install investpy
import investpy
import tensorflow as tf
from scipy.signal import argrelextrema
from tqdm.notebook import tqdm
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tqdm.keras import TqdmCallback
from datetime import datetime, timedelta
from pandas.tseries.offsets import BusinessDay

np.random.seed(1234)
tf.random.set_seed(1234)

class LSTMPipeline():
  def __init__(self, n_steps=14, units=150, activation='relu', optimizer="adam", loss=tf.keras.losses.LogCosh()):
    self.n_steps = n_steps
    self.units = units
    self.activation = activation
    self.optimizer = optimizer
    self.loss = loss
 
  def preprocessing(self, df):
    X, y = df[['USDRUB', 'Oil', 'USDCNY']].iloc[:-22], df['USDRUB'].iloc[22:] # приводим задачу к предсказанию на месяц вперед
    X, y = pd.concat([X.shift(i) for i in range(self.n_steps)], axis=1).dropna(), y.iloc[self.n_steps-1:] # генерируем лаговые переменные 
    return X, y
 
  def fit(self, X, y):
    if isinstance(X, np.ndarray) != True:
      X = X.values
    if isinstance(y, np.ndarray) != True:
      y = y.values
    self.model = Sequential()
    self.model.add(LSTM(self.units, self.activation, input_shape=(1, X.shape[1])))
    self.model.add(Dense(1))
    self.model.compile(self.optimizer, loss=self.loss)
 
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    checkpoint_filepath = '/tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                                filepath=checkpoint_filepath,
                                                save_weights_only=True,
                                                verbose=0,
                                                monitor='loss',
                                                mode='min',
                                                save_best_only=True)
    self.model.fit(X, y, epochs=500, verbose=0, callbacks=[model_checkpoint_callback])
 
    self.model.load_weights(checkpoint_filepath)
 
    return self
  
  def predict(self, X):
    if isinstance(X, np.ndarray) != True:
      X = X.values
    try:
      X = X.reshape((X.shape[0], 1, X.shape[1]))
    except:
      X = X.reshape((1, 1, X.shape[0]))

    return self.model.predict(X)

def res_return():
  oil = investpy.commodities.get_commodity_historical_data('Brent Oil', from_date='05/09/2003', to_date=datetime.strftime(datetime.today().date(), '%d/%m/%Y'))['Open']
  usdrub = investpy.currency_crosses.get_currency_cross_historical_data('USD/RUB', from_date='05/09/2003', to_date=datetime.strftime(datetime.today().date(), '%d/%m/%Y'))['Open']
  usdcny = investpy.currency_crosses.get_currency_cross_historical_data('USD/CNY', from_date='05/09/2003', to_date=datetime.strftime(datetime.today().date(), '%d/%m/%Y'))['Open']

  df = pd.concat([usdrub, oil, usdcny], axis=1).dropna()
  df.columns = ['USDRUB', 'Oil', 'USDCNY']

  df_train, df_test = df.iloc[:round(len(df)*0.8)], df.iloc[round(len(df)*0.8):]
  ltsm = LSTMPipeline(n_steps=9, units=150)

  X_train, y_train = ltsm.preprocessing(df_train)
  X_test, y_test = ltsm.preprocessing(df_test)
  
  test_usd = df['USDRUB'].iloc[df.index.isin(y_test.index)]
  X_new = pd.concat([df.iloc[-i] for i in range(1, ltsm.n_steps + 1)], axis=0)

  accuracy = []
  direction = []
  for q in tqdm(range(10)):
    ltsm.fit(X_train, y_train)
  
    pred_test = ltsm.predict(X_test)
    res = pd.concat([test_usd, pd.DataFrame(pred_test).set_index(test_usd.index)], axis=1)
    res.columns = ['True', 'Pred']

    week_day = X_test.reset_index()['Date'].dt.dayofweek.values
    total = 0
    correct = 0
    for i in range(22, len(res)):
      if week_day[i] == 0:
        if np.sign(res['Pred'].iloc[i] - res['True'].iloc[i-22]) == np.sign(res['True'].iloc[i] - res['True'].iloc[i-22]):
          correct += 1
        total += 1
    accuracy.append(correct / total)
    direction.append(np.sign(ltsm.predict(X_new)[0][0] - df['USDRUB'].iloc[-1]))
    
  return [np.sign(np.mean(direction)), np.mean(accuracy), datetime.strftime((df.index[-1] + BusinessDay(22)).date(), '%d/%m/%Y')]

res_return()