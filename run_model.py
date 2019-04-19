import torch as t
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
import pandas as pd
from scripts.models import TestModel
from scripts.trainer import Trainer
from sklearn.metrics import precision_score, recall_score
import os
import pickle

class Config:
    lr = 0.0001
    batch_size = 64
    max_epoch = 50
    gpu = True
    
opt=Config()
neutral_loss_choices = [0, 17, 18, 34, 35]
n_neutral_losses = 5
n_charges = 4

net = TestModel(input_dim=26,
                n_tasks=2*n_charges*n_neutral_losses,
                embedding_dim=256,
                hidden_dim_lstm=128,
                hidden_dim_attention=32,
                n_lstm_layers=2,
                n_attention_heads=8,
                gpu=opt.gpu)
trainer = Trainer(net, opt)

inputs = pickle.load(open('./data/massiveKB_meta.pkl', 'rb'))
samples = sorted(list(inputs.keys()))
n_samples = len(samples)
train_inputs = {k: inputs[k] for k in samples[:int(0.8*n_samples)]}
valid_inputs = {k: inputs[k] for k in samples[int(0.8*n_samples):]}

for ct in range(10):
  trainer.train(train_inputs, n_epochs=2)
  trainer.save('./models/model-%d.pth' % ct)
  sims = trainer.evaluate(valid_inputs)