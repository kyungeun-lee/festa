from libs.models.supervised import supmodel
import torch

class build_mlp(torch.nn.Module):
    def __init__(self, tasktype, input_dim, output_dim,
                 depth, width, dropout, normalization, activation,
                 optimizer, learning_rate, weight_decay):
        
        super(build_mlp, self).__init__()
        
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.tasktype = tasktype
        if normalization == "batchnorm":
            normalize_fn = torch.nn.BatchNorm1d(width)
        elif normalization == "layernorm":
            normalize_fn = torch.nn.LayerNorm(width)
        else:
            normalize_fn = torch.nn.Identity()
        
        if activation == "relu":
            act_fn = torch.nn.ReLU()
        elif activation == "lrelu":
            act_fn = torch.nn.LeakyReLU(negative_slope=0.01)
        elif activation == "sigmoid":
            act_fn = torch.nn.Sigmoid()
        elif activation == "tanh":
            act_fn = torch.nn.Tanh()
        elif activation == "gelu":
            act_fn = torch.nn.GELU()
        elif activation == "leakyrelu":
            act_fn = torch.nn.LeakyReLU()
        
        model = [torch.nn.Linear(input_dim, width), act_fn]
        for _ in range(depth-1):
            model.append(torch.nn.Linear(width, width))
            model.append(normalize_fn)
            model.append(act_fn)
            model.append(torch.nn.Dropout(dropout))
        model.append(torch.nn.Linear(width, output_dim))
        
        self.model = torch.nn.Sequential(*model)
        
    def forward(self, x, cat_features=[]):
        return self.model(x)
        
    def make_optimizer(self):
        if self.optimizer == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        
class MLP(supmodel):
    def __init__(self, params, tasktype, input_dim=0, output_dim=0, device='cuda', data_id=None, modelname="mlp"):
        
        super().__init__(params, tasktype, device, data_id, modelname)
        self.tasktype = tasktype
        self.model = build_mlp(tasktype, input_dim, output_dim, params['depth'], params['width'], params['dropout'], params['normalization'], params['activation'],
                               params['optimizer'], params['learning_rate'], params['weight_decay'])
        self.model = self.model.to(device)
    
    def fit(self, X_train, y_train):
        
        if y_train.ndim == 2:
            X_train = X_train[~torch.isnan(y_train[:, 0])]
            y_train = y_train[~torch.isnan(y_train[:, 0])]
        else:
            X_train = X_train[~torch.isnan(y_train)]
            y_train = y_train[~torch.isnan(y_train)]
            
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)
        
        super().fit(X_train, y_train)
            
    def predict(self, X_test):
        return super().predict(X_test)
        
    def predict_proba(self, X_test, logit=False):
        return super().predict_proba(X_test, logit=logit)