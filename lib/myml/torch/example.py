from torch.utils.data import Dataset

class MyDataset(Dataset):
    """
    MyDataset
    
    example)
    ```python
    data = MyDataset(X, y)
    dataloader = DataLoader(data, batch_size)
    X, y = next(iter(dataloader))
    
    ```
    """
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.X.shape[0]
    

    
    
        