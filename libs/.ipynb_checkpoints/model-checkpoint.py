from libs.transform import *
from libs.models.tree import simple_baseline
from libs.models.mlp import MLP
from libs.models.t2gformer import T2GFormer
from libs.models.fttransformer import FTTransformer
from libs.models.ae import ae
from libs.models.ict import ICT
from libs.models.meanteacher import meanteacher
from libs.models.subtab import subtab
from libs.models.scarf import SCARF
from libs.models.ssl import sslmodel
from libs.models.vime import vime
from libs.models.stunt import stunt
from libs.models.tabpfn import tabpfn
from libs.models.pseudolabel import pseudolabel
from libs.models.saint import main_saint
from libs.models.hyperfast import hyperfast

def get_model(modelname, kwargs):
    
    model_zoo = {
        "lr": lambda params: simple_baseline("lr", kwargs["tasktype"], kwargs["params"], kwargs["seed"], kwargs["cat_features"], kwargs["y_std"]),
        "knn": lambda params: simple_baseline("knn", kwargs["tasktype"], kwargs["params"], kwargs["seed"], kwargs["cat_features"], kwargs["y_std"]),
        "xgboost": lambda params: simple_baseline("xgboost", kwargs["tasktype"], kwargs["params"], kwargs["seed"], kwargs["cat_features"], kwargs["y_std"]),
        "catboost": lambda params: simple_baseline("catboost", kwargs["tasktype"], kwargs["params"], kwargs["seed"], kwargs["cat_features"], kwargs["y_std"]),
        "lightgbm": lambda params: simple_baseline("lightgbm", kwargs["tasktype"], kwargs["params"], kwargs["seed"], kwargs["cat_features"], kwargs["y_std"]),
        "mlp": lambda params: MLP(kwargs["params"], kwargs["tasktype"], kwargs["input_dim"], kwargs["output_dim"], kwargs["device"], kwargs["data_id"], "mlp"),
        "t2gformer": lambda params: T2GFormer(kwargs["n_num_features"], kwargs["cat_features"], kwargs["output_dim"]),
        "ftt": lambda params: FTTransformer(
            kwargs["params"], kwargs["tasktype"], kwargs["num_features"], kwargs["categories"], kwargs["input_dim"], kwargs["output_dim"], kwargs["device"], kwargs["data_id"], "ftt"),
        "ae": lambda params: ae(kwargs["params"], kwargs["tasktype"], kwargs["device"], 
                                transform_func=Shuffling(0.3), unsup_weight=1., data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "ict": lambda params: ICT(kwargs["params"], kwargs["tasktype"], kwargs["device"], data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "meanteacher": lambda params: meanteacher(
            kwargs["params"], kwargs["tasktype"], kwargs["device"], alpha=0.999, data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "sslsubtab": lambda params: subtab(kwargs["params"], kwargs["tasktype"], kwargs["device"], data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "sslscarf": lambda params: SCARF(kwargs["params"], kwargs["tasktype"], kwargs["device"], data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "sslmasking": lambda params: SCARF(kwargs["params"], kwargs["tasktype"], kwargs["device"], data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "sslshuffling": lambda params: SCARF(kwargs["params"], kwargs["tasktype"], kwargs["device"], data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "sslrq": lambda params: SCARF(kwargs["params"], kwargs["tasktype"], kwargs["device"], data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "sslnoisemasking": lambda params: SCARF(kwargs["params"], kwargs["tasktype"], kwargs["device"], data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "ssloriginal": lambda params: SCARF(kwargs["params"], kwargs["tasktype"], kwargs["device"], data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "sslsmallnoise": lambda params: SCARF(kwargs["params"], kwargs["tasktype"], kwargs["device"], data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "sslbinshuffling": lambda params: SCARF(kwargs["params"], kwargs["tasktype"], kwargs["device"], data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "sslbinsampling": lambda params: SCARF(kwargs["params"], kwargs["tasktype"], kwargs["device"], data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "sslrecon": lambda params: sslmodel(
            kwargs["params"], kwargs["tasktype"], kwargs["device"], Shuffling(0.3), kwargs["ssl_loss"], data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "sslbinning": lambda params: sslmodel(
            kwargs["params"], kwargs["tasktype"], kwargs["device"], Shuffling(0.3), kwargs["ssl_loss"], data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "sslvime": lambda params: sslmodel(
            kwargs["params"], kwargs["tasktype"], kwargs["device"], Shuffling(0.3), data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "semivime": lambda params: vime(
            kwargs["params"], kwargs["tasktype"], kwargs["device"], Shuffling(0.3), unsup_weight=1, data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"], num_views=3),
        "stunt": lambda params: stunt(kwargs["params"], kwargs["tasktype"], kwargs["device"], data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "tabpfn": lambda params: tabpfn(kwargs["params"], kwargs["tasktype"], kwargs["input_dim"], kwargs["output_dim"], kwargs["device"], kwargs["data_id"], "tabpfn"),
        "tabpfnv2": lambda params: tabpfn(kwargs["params"], kwargs["tasktype"], kwargs["input_dim"], kwargs["output_dim"], kwargs["device"], kwargs["data_id"], "tabpfnv2"),
        "hyperfast": lambda params: hyperfast(kwargs["params"], kwargs["tasktype"], kwargs["input_dim"], kwargs["output_dim"], kwargs["device"], kwargs["data_id"], "hyperfast"),
        "pseudolabel-masking": lambda params: pseudolabel(
            kwargs["params"], kwargs["tasktype"], kwargs["device"], ToTensor(), Masking(0.3), unsup_weight=1., data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "pseudolabel-shuffling": lambda params: pseudolabel(
            kwargs["params"], kwargs["tasktype"], kwargs["device"], ToTensor(), Shuffling(0.3), unsup_weight=1., data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "pseudolabel-noisemasking": lambda params: pseudolabel(
            kwargs["params"], kwargs["tasktype"], kwargs["device"], ToTensor(), NoiseMasking(0.3), unsup_weight=1., data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "pseudolabel-rq": lambda params: pseudolabel(
            kwargs["params"], kwargs["tasktype"], kwargs["device"], ToTensor(), RandQuant(10), unsup_weight=1., data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "pseudolabel-sampling": lambda params: pseudolabel(
            kwargs["params"], kwargs["tasktype"], kwargs["device"], ToTensor(), scarfmasking(kwargs["params"]["features_low"], kwargs["params"]["features_high"]), unsup_weight=1., data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "pseudolabel-cutmix": lambda params: pseudolabel(
            kwargs["params"], kwargs["tasktype"], kwargs["device"], ToTensor(), CutMix(kwargs["cat_features"], kwargs["num_features"]), unsup_weight=1., data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "pseudolabel-binshuffling": lambda params: pseudolabel(
            kwargs["params"], kwargs["tasktype"], kwargs["device"], ToTensor(), BinShuffling(0.3, 4, kwargs["num_features"], kwargs["params"]["boundarytype"]), 
            unsup_weight=1., data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "pseudolabel-binsampling": lambda params: pseudolabel(
            kwargs["params"], kwargs["tasktype"], kwargs["device"], ToTensor(), BinSampling(0.3, 4, kwargs["num_features"]), 
            unsup_weight=1., data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"]),
        "sslsaint": lambda params: main_saint(
            kwargs["params"], kwargs["tasktype"], kwargs["device"], data_id=kwargs["data_id"], modelname="mlp", cat_features=kwargs["cat_features"])
    }
    
    return model_zoo[modelname](kwargs)