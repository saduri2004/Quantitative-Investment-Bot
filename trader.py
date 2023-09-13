import numpy as np
import pandas as pd
import yfinance as yf

from stable_baselines import A2C
from stable_baselines.common.vec_env import DummyVecEnv

# Download data
ticker = 'AAPL'
data = yf.download(ticker, start='2017-01-01', end='2022-01-01')

# Engineer features
data['SMA10'] = data['Close'].rolling(10).mean()
data['SMA50'] = data['Close'].rolling(50).mean()

X = data[['Close','SMA10','SMA50']].values
y = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

# Custom environment 
class StockTradingEnv(DummyVecEnv):
    def __init__(self, data):
        self.data = data
        self.shape = data[['Close','SMA10','SMA50']].shape
        self.start_index = 0
        self.end_index = len(data) - 1
        
    def reset(self):
        self.start_index = np.random.randint(0, self.end_index - 10)
        return self.data.iloc[self.start_index].values
        
    def step(self, action):
        current_price = self.data.iloc[self.start_index]['Close']
        next_price = self.data.iloc[self.start_index+1]['Close']
        
        reward = 0
        done = False
        
        if action == 1 and next_price > current_price:
            reward = next_price - current_price
        elif action == 0 and next_price < current_price:
            reward = current_price - next_price
            
        self.start_index += 1
        
        if self.start_index >= self.end_index-1:
            done = True
            
        return self.data.iloc[self.start_index].values, reward, done, {}
        
# Train model       
env = StockTradingEnv(data)
model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)

# Evaluate strategy
obs = env.reset()
for i in range(len(data) - 2):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)

print("Average Reward:", np.mean(rewards))
