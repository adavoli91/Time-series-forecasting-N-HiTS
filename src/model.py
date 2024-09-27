import numpy as np
import torch
import sklearn

class Block(torch.nn.Module):
    def __init__(self, kernel_size_pool: int, expr_ratio_m1: int, dict_params: dict, num_features: int) -> None:
        '''
        Block of the N-HiTS architecture.
        
        Args:
            kernel_size_pool: Kernel size of the MaxPool layer.
            expr_ratio_m1: Inverse of the expressiveness ratio; it should be greater than 1.
            dict_params: Dictionary containing the configuration settings.
            num_features: Number of features of the input series.
            
        Returns: None.
        '''
        super().__init__()
        #
        self.len_input = dict_params['data']['len_input']
        self.horizon_forecast = dict_params['data']['horizon_forecast']
        n_neur_hidden = dict_params['model']['n_neur_hidden']
        frac_dropout = dict_params['model']['frac_dropout']
        # dimension of theta_f
        self.dim_theta_f = int(np.ceil(self.horizon_forecast/expr_ratio_m1))
        # hidden layers of the FC stack
        self.max_pool = torch.nn.MaxPool1d(kernel_size = kernel_size_pool, stride = dict_params['model']['stride_maxpool'])
        self.dense_hid_1 = torch.nn.Linear(in_features = int(np.floor((self.len_input - kernel_size_pool)/dict_params['model']['stride_maxpool'] + 1)),
                                           out_features = n_neur_hidden)
        self.dense_hid_2 = torch.nn.Linear(in_features = n_neur_hidden, out_features = n_neur_hidden)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p = frac_dropout)
        self.batch_norm_1 = torch.nn.BatchNorm1d(n_neur_hidden)
        self.batch_norm_2 = torch.nn.BatchNorm1d(n_neur_hidden)
        # dense layer for producing the theta's
        self.dense_theta_b = torch.nn.Linear(in_features = n_neur_hidden, out_features = self.len_input, bias = False)
        self.dense_theta_f = torch.nn.Linear(in_features = n_neur_hidden, out_features = self.dim_theta_f, bias = False)
        
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        y = self.max_pool(x)
        #
        y = self.dense_hid_1(y)
        y = self.batch_norm_1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.dense_hid_2(y)
        y = self.batch_norm_2(y)
        y = self.relu(y)
        y = self.dropout(y)
        # compute theta's
        theta_b = self.dense_theta_b(y)
        theta_f = self.dense_theta_f(y)
        # compute backcast
        x_hat = theta_b.reshape(*theta_b.shape, 1)
        # compute forecast
        y_hat = torch.nn.functional.interpolate(input = theta_f.reshape(*theta_f.shape, 1).transpose(1, 2), size = self.horizon_forecast).transpose(1, 2)
        #
        return x_hat, y_hat

class Stack(torch.nn.Module):
    def __init__(self, stack_number: int, dict_params: dict, num_features: int) -> None:
        '''
        Stack of the N-HiTS architecture.
        
        Args:
            stack_number: Integer indicating the block index in the stack, starting from 1.
            dict_params: Dictionary containing the configuration settings.
            num_features: Number of features of the input series.
            
        Returns: None
        '''
        super().__init__()
        stack_params = dict_params['model'][f'stack_{stack_number}']
        kernel_size_pool = stack_params['kernel_size_pool']
        expr_ratio_m1 = stack_params['expr_ratio_m1']
        #
        self.block = Block(kernel_size_pool = kernel_size_pool, expr_ratio_m1 = expr_ratio_m1, dict_params = dict_params,
                           num_features = num_features)
        
    def forward(self, x):
        x_hat, y = self.block(x)
        x = x - x_hat
        #
        x_hat = x
        y_hat = y
        return x_hat, y_hat

class NHiTS(torch.nn.Module):
    def __init__(self, dict_params: dict, num_features: int) -> None:
        '''
        N-HiTS architecture.
        
        Args:
            dict_params: Dictionary containing the configuration settings.
            num_features: Number of features of the input series.
            
        Returns: None
        '''
        super().__init__()
        #
        self.stack_1 = Stack(stack_number = 1, dict_params = dict_params, num_features = num_features)
        self.stack_2 = Stack(stack_number = 2, dict_params = dict_params, num_features = num_features)
        self.stack_3 = Stack(stack_number = 3, dict_params = dict_params, num_features = num_features)
        
    def forward(self, x):
        # first stack
        x_hat, y_hat_1 = self.stack_1(x)
        x = x - x_hat
        # second stack
        x_hat, y_hat_2 = self.stack_2(x)
        x = x - x_hat
        # third stack
        _, y_hat_3 = self.stack_3(x)
        #
        return y_hat_1, y_hat_2, y_hat_3
    
class TrainNHiTS:
    def __init__(self, model: torch.nn.Module, dict_params: dict, dataloader_train: torch.utils.data.DataLoader, dataloader_valid: torch.utils.data.DataLoader):
        '''
        Class to train the N-HiTS model.
        
        Args:
            model: PyTorch model.
            dict_params: Dictionary containing information about the model architecture.
            dataloader_train: Dataloader containing training data.
            dataloader_valid: Dataloader containing validation data.
            
        Returns: None.
        '''
        self.model = model
        self.dict_params = dict_params
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(params = model.parameters())
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = self.optimizer, factor = 0.5)
        self.dataloader_train = dataloader_train
        self.dataloader_valid = dataloader_valid

    def _model_on_batch(self, batch: tuple, training: bool, loss_epoch: float) -> float:
        '''
        Function to perform training on a single batch of data.
        
        Args:
            batch: Batch of data to use for training/evaluation.
            training: Whether to perform training (if not, evaluation is understood).
            loss_epoch: Loss of the current epoch.
            
        Returns:
            loss: Value of the loss function.
        '''
        if training == True:
            self.optimizer.zero_grad()
        #
        x, y_true = batch
        x = x.to('cpu')
        y_true = y_true.to('cpu')
        y_hat_1, y_hat_2, y_hat_3 = self.model(x)
        y_hat_1 = y_hat_1.to('cpu')
        y_hat_2 = y_hat_2.to('cpu')
        y_hat_3 = y_hat_3.to('cpu')
        #
        loss = self.loss_func(y_hat_1 + y_hat_2 + y_hat_3, y_true)
        #
        if training == True:
            loss.backward()
            self.optimizer.step()
        #
        return loss.item()

    def _train(self) -> float:
        '''
        Function to train the TCN model on a single epoch.
        
        Args: None.
            
        Returns:
            loss: Value of the training loss function per batch.
        '''
        self.model.train()
        loss_epoch = 0
        for batch in self.dataloader_train:
            loss_epoch += self._model_on_batch(batch = batch, training = True, loss_epoch = loss_epoch)
        return loss_epoch/len(self.dataloader_train)

    def _eval(self) -> float:
        '''
        Function to evaluate the TCN model on the validation set on a single epoch.
        
        Args: None.
            
        Returns:
            loss: Value of the validation loss function per batch.
        '''
        self.model.eval()
        loss_epoch = 0
        with torch.no_grad():
            for batch in self.dataloader_valid:
                loss_epoch += self._model_on_batch(batch = batch, training = False, loss_epoch = loss_epoch)
        return loss_epoch/len(self.dataloader_valid)

    def train_model(self) -> (torch.nn.Module, list, list):
        '''
        Function to train the TCN model.
        
        Args: None.
            
        Returns:
            model: Trained N-HiTS model.
            list_loss_train: List of training loss function across the epochs.
            list_loss_valid: List of validation loss function across the epochs.
        '''
        dict_params = self.dict_params
        n_epochs = dict_params['training']['n_epochs']
        list_loss_train, list_loss_valid = [], []
        counter_patience = 0
        for epoch in range(1, n_epochs + 1):
            loss_train = self._train()
            loss_valid = self._eval()
            #
            if (len(list_loss_valid) > 0) and (loss_valid >= np.min(list_loss_valid)*(1 - dict_params['training']['min_delta_loss_perc'])):
                counter_patience += 1
            if (len(list_loss_valid) == 0) or ((len(list_loss_valid) > 0) and (loss_valid < np.min(list_loss_valid))):
                counter_patience = 0
                torch.save(self.model.state_dict(), '../data/artifacts/weights.p')
            if counter_patience >= dict_params['training']['patience']:
                print(f'Training stopped at epoch {epoch}. Restoring weights from epoch {np.argmin(list_loss_valid) + 1}.')
                self.model.load_state_dict(torch.load('../data/artifacts/weights.p'))
                break
            #
            print(f'Epoch {epoch}: training loss = {loss_train:.4f}, validation loss = {loss_valid:.4f}, patience counter = {counter_patience}.')
            self.scheduler.step(loss_valid)
            #
            list_loss_train.append(loss_train)
            list_loss_valid.append(loss_valid)
        if epoch == n_epochs:
            self.model.load_state_dict(torch.load('../data/artifacts/weights.p'))
        return self.model, list_loss_train, list_loss_valid
    
def get_y_true_y_hat(model: torch.nn.Module, x: torch.tensor, y: torch.tensor, date_y: np.array,
                     scaler: sklearn.preprocessing.StandardScaler) -> (np.array, np.array, np.array):
    '''
    Function to get the real time series and its prediction.
    
    Args:
        model: Trained N-BEATS model.
        x: Tensor representing regressors.
        y: Tensor representing target time series.
        date_y: Array containing the dates corresponding to the elements of `y`.
        scaler: Scaled used to rescale data.
        
    Returns:
        y_true: Array containing the true values.
        y_hat_trend: Array containing the predicted trend.
        y_hat_seas: Array containing the predicted seasonality.
    '''
    list_date = []
    y_true = []
    y_hat_1, y_hat_2, y_hat_3 = [], [], []
    pred_1, pred_2, pred_3 = model(x)
    for i in range(np.unique(date_y).shape[0]):
        date = np.unique(date_y)[i]
        list_date.append(date)
        idx = np.where(date_y == date)
        y_true.append(y.numpy()[idx].mean())
        y_hat_1.append(pred_1.detach().numpy()[idx].mean())
        y_hat_2.append(pred_2.detach().numpy()[idx].mean())
        y_hat_3.append(pred_3.detach().numpy()[idx].mean())
    y_true = np.array(y_true)
    y_hat_1 = np.array(y_hat_1)
    y_hat_2 = np.array(y_hat_2)
    y_hat_3 = np.array(y_hat_3)
    # scale back
    y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
    y_hat_1 = scaler.inverse_transform(y_hat_1.reshape(-1, 1)).ravel()
    y_hat_2 = scaler.inverse_transform(y_hat_2.reshape(-1, 1)).ravel()
    y_hat_3 = scaler.inverse_transform(y_hat_3.reshape(-1, 1)).ravel()
    #
    return y_true, y_hat_1, y_hat_2, y_hat_3

def compute_mape(y_true: np.array, y_hat_1: np.array, y_hat_2: np.array, y_hat_3: np.array, scaler: sklearn.preprocessing.StandardScaler) -> float:
    '''
    Function to compute the MAPE.
    
    Args:
        y_true: Array containing the true values.
        y_hat_1: Array containing the first predicted component.
        y_hat_2: Array containing the second predicted component.
        y_hat_3: Array containing the third predicted component.
        scaler: Scaled used to rescale data.
        
    Returns:
        mape: MAPE computed from `y_true` and `y_hat`.
    '''
    y_hat = y_hat_1 + y_hat_2 + y_hat_3 - 2*scaler.mean_
    mape = np.mean(abs(y_true[y_true > 0] - y_hat[y_true > 0])/y_true[y_true > 0])
    return mape