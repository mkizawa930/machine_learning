import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


   


def preprocessing(df: pd.DataFrame):
    df['y'] = np.log(df['Adj Close']).diff().shift(-1)
    columns = ['Adj Close', 'High', 'Low', 'Open', 'Volume']
    # scaling
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(df.loc[[*columns, 'y']])
    
    train, test = train_test_split(scaled_df, test_size=0.2)
    return train, test
    
    