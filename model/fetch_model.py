import torch
from model.model import *

def fetch_model(model_name, cfg):
    # Previous models
    if model_name == "hand_occ_net":
        model = HandOccNet(cfg)
    elif model_name == "semi_hand":
        model = SemiHand(cfg)
    elif model_name == "mobrecon":
        model = MobRecon_DS(cfg)
    elif model_name == "h2onet":
        model = H2ONet(cfg)
    elif model_name == "simple_hand":
        model = SimpleHand(cfg)
    elif model_name == "semi_hand_object":
        model = SemiHandObject(cfg)
    elif model_name == "kypt_transformer":
        model = KyptTransformer(cfg)
    elif model_name == "hflnet":
        model = HFLNet(cfg)
    
    # UniHOPE model
    elif model_name == "unihope_net":
        model = UniHOPENet(cfg)
    
    # Pre-trained classifiers
    elif model_name == "classifier":
        model = GraspClassifier(cfg)
    
    # Classifier based two-in-one models (A+B)
    elif model_name == "twoinone_baseline_classifier":
        model = TwoInOneBaselineClassifier(cfg)
    
    return model