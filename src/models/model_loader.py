import xgboost
import pickle
from pytorch_tabnet.tab_model import TabNetClassifier

def load_pickled_model(fname):
    return pickle.load(open(fname+".save", "rb"))

def load_trained_tabnet(fname):
    loaded_clf = TabNetClassifier()
    loaded_clf.load_model(fname)
    return loaded_clf

def load_trained_xgboost(fname):
    loaded_clf = xgboost.XGBClassifier()
    loaded_clf.load_model(fname+".json")
    return loaded_clf