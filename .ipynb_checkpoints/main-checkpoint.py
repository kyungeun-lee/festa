import torch, os, torchvision, datetime, time, argparse, logging, json
from libs.data import TabularDataset
from libs.utils import *
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, r2_score
import pandas as pd
from libs.model import *
import time
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger("lightgbm").setLevel(logging.WARNING)

parser = argparse.ArgumentParser()

parser.add_argument("--gpu_id", type=int, default=4)
parser.add_argument("--openml_id", type=int, default=4538)
parser.add_argument("--shot", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--force_train", action="store_true")

parser.add_argument("--config_filename", type=str, default="sslbinning.yaml")

args = parser.parse_args()

configs = load_config(args.config_filename, shot=args.shot)
modelname = configs["modelname"]
savepath = f'results/seed={args.seed}/shot={args.shot}/model={modelname}/data={args.openml_id}/'

if not os.path.exists(savepath):
    os.makedirs(savepath)
with open(f'{savepath}/config.yaml', 'w') as f:
    yaml.dump(configs, f)

def main():   
    
    train = check_trained(savepath)
    if train:
        log = logging.getLogger()
        log.setLevel(logging.INFO)
        file_handler = logging.FileHandler(os.path.join(savepath, 'logs.log'))
        log.addHandler(file_handler)
        log.addHandler(TqdmLoggingHandler())
        log.info("Results will be saved at.. %s" %savepath)

        torch.cuda.set_device(args.gpu_id)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        env_info = '{0}:{1}'.format(os.uname().nodename, args.gpu_id)
        log.info(env_info)
        
        with open(f'dataset_id.json', 'r') as file:
            data_info = json.load(file)
        tasktype = data_info.get(str(args.openml_id))['tasktype']
        print(tasktype)
        
        dataset = TabularDataset(args.openml_id, tasktype, device, labeled_data=args.shot,
                                 seed=args.seed, modelname=configs['modelname'])
        (X_train, y_train), (X_test, y_test) = dataset._indv_dataset()
        labeled_idx = torch.where(~y_train.isnan())[0].cpu().numpy()
        print(f'Full train data size: {X_train.size(0)} {y_train.size()}')
        
        try:
            configs["params"]["input_dim"] = X_train.size(1)
            configs["params"]["output_dim"] = y_train.size(1) if tasktype == "multiclass" else 1
            configs["params"]["shots"] = args.shot
            configs["params"]["features_low"] = X_train.min(axis=0).values.cpu().numpy()
            configs["params"]["features_high"] = X_train.max(axis=0).values.cpu().numpy()   
            epsilon = 1e-10
            configs["params"]["features_high"] = np.where(
                configs["params"]["features_low"] == configs["params"]["features_high"],
                configs["params"]["features_high"] + epsilon, configs["params"]["features_high"])
            configs["params"]["num_features"] = dataset.X_num
            configs["params"]["data_id"] = args.openml_id
            configs["params"]["gpu_id"] = str(args.gpu_id)
            configs["params"]["categories"] = dataset.X_categories
            if modelname != "catboost":
                configs["params"]["cat_features"] = dataset.X_cat
            configs["params"]["dim"] = 32 if X_train.size(1) < 20 else 8 ## reference: SAINT (Appendix C and dataset stats)
        except TypeError:
            pass
        
        kwargs = dict({
            "tasktype": tasktype,
            "params": configs["params"], 
            "seed": args.seed, 
            "cat_features": dataset.X_cat, 
            "y_std": dataset.y_std,
            "input_dim": X_train.size(1), 
            "output_dim": y_train.size(1) if tasktype == "multiclass" else 1,
            "device": device, 
            "data_id": args.openml_id,
            "num_features": dataset.X_num, 
            "categories": dataset.X_categories,
            "ssl_loss": configs.get("ssl_loss", None)})
        
        try:
            model = get_model(modelname, kwargs)
        except KeyError:
            raise ValueError(f'check the model name ({modelname})')
        
        print("Start fitting")
        st = time.time()
        model.fit(X_train, y_train)
        if modelname in ["stunt"]:
            train_preds, test_preds, train_prob, test_prob = None, None, None, None
        else:
            try:
                test_preds = model.predict(X_test)
                test_prob = model.predict_proba(X_test)
            except torch.OutOfMemoryError:
                test_preds = []; test_prob = []
                for j in range(X_test.shape[0] // 100 + 1):
                    test_preds.append(model.predict(X_test[j*100:(j+1)*100]))
                    test_prob.append(model.predict_proba(X_test[j*100:(j+1)*100]))
                test_preds = np.concatenate(test_preds, axis=0)
                test_prob = np.concatenate(test_prob, axis=0)
        
        if modelname.startswith("ssl"):
            train_score = dict(); test_score = dict()
            for i, evalmethod in enumerate(["lr", "knn", "lineareval", "finetuning"]):
                train_score[evalmethod] = evaluate(train_preds[i][labeled_idx], y_train[labeled_idx], train_prob[i][labeled_idx], tasktype=tasktype, y_std=dataset.y_std)
                test_score[evalmethod] = evaluate(test_preds[i], y_test, test_prob[i], tasktype=tasktype, y_std=dataset.y_std)
        elif modelname == "stunt":
            train_score = model.evaluate(X_train, y_train, X_train, y_train)
            test_score = model.evaluate(X_train, y_train, X_test, y_test)
        elif (modelname in ["tabpfn"]) & (train_preds is not None):
            train_score = evaluate(train_preds[labeled_idx], y_train[labeled_idx], train_prob[labeled_idx], tasktype=tasktype, y_std=dataset.y_std, tabpfn=True)
            test_score = evaluate(test_preds, y_test, test_prob, tasktype=tasktype, y_std=dataset.y_std, tabpfn=True)
        elif (modelname == "tabpfn"): ## not applicable datasets
            train_score = [0., 0.]; test_score = [0., 0.]
        else:
            train_score = evaluate(train_preds[labeled_idx], y_train[labeled_idx], train_prob[labeled_idx], tasktype=tasktype, y_std=dataset.y_std)
            test_score = evaluate(test_preds, y_test, test_prob, tasktype=tasktype, y_std=dataset.y_std)
        
        saveresults(
            modelname, savepath, train_preds, test_preds, train_prob,
            test_prob, tasktype, train_score, test_score)
        print(test_score)

if __name__ == "__main__":   
    main()
    print("==================================================================")
