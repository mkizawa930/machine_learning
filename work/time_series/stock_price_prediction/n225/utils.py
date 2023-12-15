import numpy as np
import pandas as pd
from typing import List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


   
from torch.utils.data import DataLoader


def preprocessing(df: pd.DataFrame, columns: List[str]):
    # scaling
    scaler = StandardScaler()
    scaled_df = df.copy()
    scaled_df.loc[:, columns] = scaler.fit_transform(df.loc[:, columns].values)
    
    # train, test = train_test_split(scaled_df, test_size=0.2, shuffle=False)
    return scaled_df
    
    
class Trainer:
    """
    Trainer
    """
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion
        
    def train(self, epochs, train_dataset, val_dataset, batch_size):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        for epoch in range(epochs):
            train_loss = .0
            val_loss = .0
            
            for x, y in train_dataloader:
                loss = self._train_step(x, y)
                train_loss += loss.item()
                
            trian_loss /= batch_size
            
            for x, y in val_dataloader:
                _, loss = self._val_step(x, y)
                val_loss += loss.item()
            
            val_loss /= batch_size
            
            print(f'epoch: {epoch} train loss: {train_loss}, val loss: {val_loss}')
            
    
    def _compute_loss(self, y, yhat):
        return self.criterion(yhat, y)     
            
    def _train_step(self, x, y):
        self.model.train()
        yhat = self.model(x)
        loss = self._compute_loss(y, yhat)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return yhat, loss
    
    def _val_step(self, x, y):
        self.model.eval()
        yhat = self.model(x)
        loss = self._compute_loss(y, yhat)
        return yhat, loss
    
    
    
        