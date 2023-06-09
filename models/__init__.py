
import os
import torch
from importlib import import_module
import torch.nn as nn
def init_model(model, args):

    device = torch.device(args.device)
    model.to(device)


    return model#, optimizer, loss_scaler

def get_model(args):
    module = import_module('models.' + args.model.lower())
    return init_model(module.make_model(args), args)
