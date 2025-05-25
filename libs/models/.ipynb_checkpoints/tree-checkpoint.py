import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import Pool, CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from libs.models.supervised import filter_params

class simple_baseline:
    def __init__(self, modelname, tasktype, params, seed, cat_features=[], y_std=None):
        
        self.modelname = modelname
        self.tasktype = tasktype
        self.cat_features = cat_features
        self.y_std = y_std
        
        if modelname == "xgboost":
            pred_fn = {"multiclass": xgb.XGBClassifier, "binclass": xgb.XGBClassifier, "regression": xgb.XGBRegressor}
            loss_fn = {"multiclass": "multi:softmax", "binclass": "binary:logistic", "regression": "reg:squarederror"}
            params = filter_params(pred_fn[tasktype], params)
            self.model = pred_fn[tasktype](
                tree_method='gpu_hist', objective=loss_fn[tasktype],
                gpu_id=0, booster='gbtree', random_state=seed, **params
            )
        elif modelname == 'catboost':            
            pred_fn = {"multiclass": CatBoostClassifier, "binclass": CatBoostClassifier, "regression": CatBoostRegressor}
            loss_fn = {"multiclass": "MultiClass", "binclass": "CrossEntropy", "regression": "RMSE"}
            if params["data_id"] == 1492:
                params["task_type"] = "GPU"
                params["devices"] = params["gpu_id"]
            params = filter_params(pred_fn[tasktype], params)
            self.model = pred_fn[tasktype](
                loss_function=loss_fn[tasktype], cat_features=cat_features, random_state=seed, verbose=0, **params)
        elif modelname == 'lightgbm':
            loss_fn = {"multiclass": "multiclass", "binclass": "binary", "regression": "regression"}
            model_fn = {"multiclass": LGBMClassifier, "binclass": LGBMClassifier, "regression": LGBMRegressor}
            params = filter_params(model_fn[tasktype], params)
            self.model = model_fn[tasktype](objective=loss_fn[tasktype], verbose=-1, verbose_eval=False, random_state=seed, **params)
        elif modelname == "lr":
            self.model = LogisticRegression(random_state=seed)
        elif modelname == "knn":
            self.model = KNeighborsClassifier(n_neighbors=params["k"])        

    def fit(self, X_train, y_train):
        X_train = X_train.cpu().numpy()
        y_train = y_train.cpu().numpy()
        
        labeled_flag = np.unique(np.where(~np.isnan(y_train))[0])
        X_train = X_train[labeled_flag]
        y_train = y_train[labeled_flag]
        
        if self.tasktype == "multiclass":
            y_train = np.argmax(y_train, axis=1)
            
        if self.modelname == "catboost":
            X_train = pd.DataFrame(X_train).astype({k: 'int' for k in self.cat_features})
            dtrain = Pool(X_train, label=y_train, cat_features=self.cat_features)
        
        if self.modelname == 'catboost':
            return self.model.fit(dtrain)
        elif self.modelname == "lightgbm":
            return self.model.fit(X_train, y_train, categorical_feature="auto")
        else:
            return self.model.fit(X_train, y_train)
        
    def predict(self, X_test):
        X_test = X_test.cpu().numpy()
        if self.modelname in ['catboost', 'lightgbm']:
            X_test = pd.DataFrame(X_test).astype({k: 'int' for k in self.cat_features})
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test, logit=False):
        X_test = X_test.cpu().numpy()
        if self.modelname in ['catboost', 'lightgbm']:
            X_test = pd.DataFrame(X_test).astype({k: 'int' for k in self.cat_features})
        if self.tasktype in ['binclass', 'multiclass']:
            return self.model.predict_proba(X_test)
        else:        
            return None