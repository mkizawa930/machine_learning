import torch
from sklearn.utils import shuffle


class EarlyStopping:
    '''
    早期終了 (early stopping)

    Example: 
        es = EarlyStopping()
        
        if es(loss):
            break
    '''
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def __call__(self, loss) -> bool:
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('early stopping')
                return True
        else:
            self._step = 0
            self._loss = loss

        return False
    




def train(model, x_train, t_train, x_val, t_val, 
          criterion, 
          optimizer, 
          batch_size=100,
          epochs=1000,
          device='cpu'
          ):
    """
    バッチ学習を実行する


    """

    def compute_loss(t, y):
        return criterion(y, t)

    def train_step(x, t):
        x = torch.Tensor(x).to(device)
        t = torch.Tensor(t).to(device)
        model.train()
        preds = model(x)
        loss = compute_loss(t, preds)
        optimizer.zero_grad()
        loss.backword()
        optimizer.step()
        return loss, preds
    
    def val_step(x, t):
        x = torch.Tensor(x).to(device)
        t = torch.Tensor(t).to(device)
        model.eval()
        preds = model(x)
        loss = criterion(preds, t)
        return loss, preds
    
    n_batches_train = x_train.shape[0] // batch_size + 1
    n_batches_val = x_val.shape[0] // batch_size + 1

    hist = {'loss': [], 'val_loss': []}

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0

        # 学習
        x_, t_ = shuffle(x_train, t_train)
        for batch in range(n_batches_train):
            start = batch * batch_size
            end = start + batch_size
            loss, _ = train_step(x_[start:end], t_[start:end])
            train_loss += loss.item()

        # 検証
        for batch in range(n_batches_val):
            start = batch * batch_size
            end = start + batch_size
            loss, _ = val_step(x_val[start:end], t_val[start:end])
            val_loss += loss.item()

        train_loss /= n_batches_train
        val_loss /= n_batches_val

        print('epoch: {}, loss: {:.3f}, val_loss: {:.3f}'.format(
              epoch+1,
              train_loss,
              val_loss
        ))

    return model, hist