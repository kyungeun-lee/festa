from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import openml, torch
import numpy as np
import pandas as pd
import sklearn.datasets
import scipy.stats
from sklearn.preprocessing import QuantileTransformer

def load_data(openml_id):
    if openml_id == 999999:
        dataset = sklearn.datasets.fetch_california_housing()
        X = pd.DataFrame(dataset['data'])
        y = pd.DataFrame(dataset['target'])
    elif openml_id == 43611:
        dataset = openml.datasets.get_dataset(openml_id)
        print(f'Dataset is loaded.. Data name: {dataset.name}, Target feature: class')
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target="class"
        )
    else:
        dataset = openml.datasets.get_dataset(openml_id)
        print(f'Dataset is loaded.. Data name: {dataset.name}, Target feature: {dataset.default_target_attribute}')
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute
        )
    
    if openml_id == 537:
        y = y / 10000
    
    nan_counts = X.isna().values.sum()
    cell_counts = X.shape[0] * X.shape[1]
    n_samples = X.shape[0]
    n_cols = X.shape[1]
    ### NaN 처리
    #1. column -- 50% 이상이 nan인 경우 column 탈락
    nan_cols = X.isna().sum(0)
    valid_cols = nan_cols.loc[nan_cols < (0.5*len(X))].index.tolist()
    X = X[valid_cols]
    #2. remained nans -> delete rows
    nan_idx = X.isna().any(axis=1)
    X = X[~nan_idx].reset_index(drop=True)
    y = y[~nan_idx].reset_index(drop=True)
    
    for col in X.select_dtypes(exclude=['float', 'int']).columns:
        colencoder = LabelEncoder()
        X[col] = colencoder.fit_transform(X[col])
    
    y = y.values
    if isinstance(y[0], str):
        labelencoder = LabelEncoder()
        y = labelencoder.fit_transform(y)
    
    return X.values, y, attribute_names, dataset.default_target_attribute

def one_hot(y):
    num_classes = len(np.unique(y))
    min_class = y.min()
    enc = LabelEncoder()
    y_ = enc.fit_transform(y - min_class)
    return np.eye(num_classes)[y_]

def split_data(X, y, tasktype, seed=123456, device='cuda'):
    
    if tasktype == "multiclass":
        y = one_hot(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=seed) ## same as STUNT
    
    X_train = torch.from_numpy(X_train).type(torch.float32).to(device)
    X_test = torch.from_numpy(X_test).type(torch.float32).to(device)

    y_train = torch.from_numpy(y_train).type(torch.float32).to(device)
    y_test = torch.from_numpy(y_test).type(torch.float32).to(device)
    
    return (X_train, y_train), (X_test, y_test)

def cat_num_features(X_train, cat_threshold=20):
    num_features = X_train.shape[1]
    
    counts = torch.tensor([X_train[:, i].unique().numel() for i in range(num_features)])
    if cat_threshold == None:
        X_cat = []
        X_num = np.arange(num_features)
    else:
        X_cat = np.where(counts <= cat_threshold)[0].astype(int)
        X_num = np.array([int(i) for i in range(num_features) if not i in X_cat])
    return (X_cat, counts[X_cat], X_num)

def standardization(X, X_mean, X_std, y, y_mean=0, y_std=1, num_indices=[], tasktype='multiclass'):
    X[:, num_indices] = (X[:, num_indices] - X_mean[num_indices]) / (X_std[num_indices] + 1e-10)
#     if tasktype == "regression":
#         y = (y - y_mean) / (y_std + 1e-10)
    return (X, y)

def quant(X_train, X_test, y_train, y_test, y_mean=0, y_std=1, num_indices=[], tasktype='multiclass'):
    device = X_train.get_device()
    quantile_transformer = QuantileTransformer(output_distribution='uniform', random_state=42)
    X_train[:, num_indices] = torch.tensor(quantile_transformer.fit_transform(X_train[:, num_indices].cpu().numpy()), device=device)
    X_test[:, num_indices] = torch.tensor(quantile_transformer.transform(X_test[:, num_indices].cpu().numpy()), device=device)
#     if tasktype == "regression":
#         y_train = (y_train - y_mean) / (y_std + 1e-10)
#         y_test = (y_test - y_mean) / (y_std + 1e-10)
    return (X_train, y_train), (X_test, y_test)


class TabularDataset(torch.utils.data.Dataset):
    def __init__(self, openml_id, tasktype, device, labeled_data=0,
                 cat_threshold=20, seed=123456, modelname="xgboost", normalize=True, quantile=True):
        X, y, attribute_names, target_name = load_data(openml_id)
            
        self.tasktype = tasktype
        self.attribute_names = attribute_names
        self.target_name = target_name
        self.raw_ys = np.unique(y)
        
        (self.X_train, self.y_train), (self.X_test, self.y_test) = split_data(X, y, self.tasktype, seed=seed)
        (self.X_cat, self.X_categories, self.X_num) = cat_num_features(torch.tensor(X), cat_threshold=cat_threshold)
        if modelname in ["ftt", "resnet", "t2gformer", "catboost", "lightgbm", "sslsaint"]: ### including special modules for categorical features
            for cat_dim in self.X_cat:
                unique_values = torch.cat([self.X_train[:, cat_dim].unique(), self.X_test[:, cat_dim].unique()]).unique() 
                mapping = {v.item(): idx for (idx, v) in enumerate(unique_values)}
                self.X_train[:, cat_dim] = torch.tensor([mapping[v.item()] for v in self.X_train[:, cat_dim]])
                self.X_test[:, cat_dim] = torch.tensor([mapping[v.item()] for v in self.X_test[:, cat_dim]])
        else:
            for cat_dim in self.X_cat:
                self.X_train[:, cat_dim] = self.X_train[:, cat_dim] / cat_threshold
                self.X_test[:, cat_dim] = self.X_test[:, cat_dim] / cat_threshold
                
        print("input dim: %i, cat: %i, num: %i" %(self.X_train.size(1), len(self.X_cat), len(self.X_num)))
        
        ### get small data here
        torch.manual_seed(seed)
        if labeled_data > 0:
            if tasktype in ['binclass', 'multiclass']:
                unique_classes, full_indices, n_min_samples = self.y_train.unique(dim=0, return_inverse=True, return_counts=True)
                num_classes = len(unique_classes)
                n_data = len(self.X_train)
                
                n_samples_per_class = [labeled_data] * num_classes                
                assert n_min_samples.min() > max(n_samples_per_class), "check the class imbalance!"

                indices = []
                for i, class_idx in enumerate(unique_classes):
                    class_indices = torch.where(full_indices == i)[0]
                    indices.append(class_indices[torch.randperm(len(class_indices))][:n_samples_per_class[i]])
                subsample_indices = torch.cat(indices)
            else:
                n_data = len(self.X_train)
                subsample_indices = torch.randperm(n_data)[:labeled_data]
            
            unlabeled_indices = [i for i in range(n_data) if not i in subsample_indices]
            self.y_train[unlabeled_indices] = torch.nan        
        
        self.batch_size = 100 ##ref: STUNT
        self.X_mean = self.X_train.mean(0)
        self.X_std = self.X_train.std(0)
        self.y_mean = self.y_train.type(torch.float).mean(0)
        self.y_std = self.y_train.type(torch.float).std(0)
                
        if quantile & (len(self.X_num) > 0) & normalize:
            (self.X_train, self.y_train), (self.X_test, self.y_test) = quant(
                self.X_train, self.X_test, self.y_train, self.y_test,
                self.y_mean, self.y_std, num_indices=self.X_num, tasktype=self.tasktype)
        elif (not quantile) & normalize:
            (self.X_train, self.y_train) = standardization(self.X_train, self.X_mean, self.X_std, self.y_train, self.y_mean, self.y_std, num_indices=self.X_num, tasktype=self.tasktype)
            (self.X_test, self.y_test) = standardization(self.X_test, self.X_mean, self.X_std, self.y_test, self.y_mean, self.y_std, num_indices=self.X_num, tasktype=self.tasktype)
            
        
            
    def __len__(self, data):
        if data == "train":
            return len(self.X_train)
        else:
            return len(self.X_test)
    
    def _indv_dataset(self):
        return (self.X_train, self.y_train), (self.X_test, self.y_test)
    
    def __getitem__(self, idx, data):
        if data == "train":
            return self.X_train[idx], self.y_train[idx]
        else:
            return self.X_test[idx], self.y_test[idx]
        
from sklearn.preprocessing import KBinsDiscretizer

def Binning(dataset, num_bins, device, binning_reg=True):
    
    binmodel = KBinsDiscretizer(n_bins=num_bins, encode="ordinal", strategy="quantile")
    X_binned = binmodel.fit_transform(dataset.cpu().numpy())

    if binning_reg:
        bin_edges = binmodel.bin_edges_
        
        bin_means = [0] * len(bin_edges)
        for i in range(len(bin_edges)):
            bin_means[i] = [(bin_edges[i][j] + bin_edges[i][j + 1]) / 2 for j in range(len(bin_edges[i]) - 1)]

        X_binned_means = np.zeros_like(X_binned, dtype=float)
        for i in range(X_binned.shape[0]):
            for j in range(X_binned.shape[1]):
                if len(np.unique(X_binned[:, j])) == 1:
                    X_binned_means[i, j] = 0.5
                else:
                    X_binned_means[i, j] = bin_means[j][int(X_binned[i, j])]
        
        return torch.tensor(X_binned_means, device=device)
    else:
        return torch.tensor(X_binned, device=device)
    