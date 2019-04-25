#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 17:48:03 2018

@author: zqwu
"""


import torch as t
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from copy import deepcopy
import numpy as np
from sklearn.metrics import r2_score
from scripts.models import CELoss
from torch.utils.data import Dataset, DataLoader
import os
import pickle

class Trainer(object):
  def __init__(self, 
               net, 
               opt, 
               criterion=CELoss,
               featurize=True):
    self.net = net
    self.opt = opt
    self.criterion = criterion
    if self.opt.gpu:
      self.net = self.net.cuda()
  
  def save(self, path):
    t.save(self.net.state_dict(), path)
  
  def load(self, path):
    s_dict = t.load(path, map_location=lambda storage, loc: storage)
    self.net.load_state_dict(s_dict)

  def set_seed(self, seed):
    t.manual_seed(seed)
    if self.opt.gpu:
      t.cuda.manual_seed_all(seed)

  def train(self, train_data, n_epochs=None, **kwargs):
    self.run_model(train_data, train=True, n_epochs=n_epochs, **kwargs)
    return
  
  def display_loss(self, train_data, **kwargs):
    self.run_model(train_data, train=False, n_epochs=1, **kwargs)
    return
    
  def run_model(self, data_mapping, train=False, n_epochs=None, log_every_step=1000, **kwargs):
    if train:
      optimizer = Adam(self.net.parameters(),
                       lr=self.opt.lr,
                       betas=(.9, .999))
      self.net.zero_grad()
      epochs = self.opt.max_epoch
    else:
      epochs = 1
    if n_epochs is not None:
      epochs = n_epochs
    n_points = len(data_mapping)
    
    dataset = SpectraDataset(data_mapping, self.opt.batch_size, train=True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    for epoch in range(epochs):
      epoch_loss = 0.
      print ('start epoch {epoch}'.format(epoch=epoch))
      ct = 0
      loss = 0.
      for batch in data_loader:
        ###
        # batch[0]: X,      seq_len * batch_size * input_dim{26}
        # batch[1]: X_meta, batch_size * meta_input_dim{2}
        # batch[2]: y,      (seq_len - 1) * batch_size * (2 * n_charges{4}) * n_neutral_losses{5}
        # batch[3]: w,      batch_size
        ###
        X, X_metas, y, weights = batch[0][0], batch[1][0], batch[2][0], batch[3][0]
        if self.opt.gpu:
          X = X.cuda()
          X_metas = X_metas.cuda()
          y = y.cuda()
          weights = weights.cuda()
        
        output = self.net(X, X_metas, self.opt.batch_size)
        error = self.criterion(y, output, self.opt.batch_size)
        loss += error
        epoch_loss += error
        ct += 1
        error.backward()
        if train:
          optimizer.step()
          self.net.zero_grad()
        if ct > 0 and ct%log_every_step == 0:
          print ('\tepoch {epoch}, {ct} steps, loss: {loss}'.format(epoch=epoch, ct=ct, loss=loss.item()/log_every_step/self.opt.batch_size))
          loss = 0.
      print ('epoch {epoch} loss: {loss}'.format(epoch=epoch, loss=epoch_loss.item()/n_points))
      
  def predict(self, test_data_mapping, log_every_step=100000):
    n_points = len(test_data_mapping)
    #test_data = self.load_data_from_files(test_data_mapping)
    dataset = SpectraDataset(test_data_mapping, 1, train=True)
    test_preds = []
    
    inputs = [[], []]
    labels = []
    for i, batch in enumerate(dataset):
      if i > 0 and i%log_every_step == 0:
        print("Finished %d samples" % i)
      X, X_metas, y = batch[0], batch[1], batch[2]
      labels.append(y[:, 0].cpu().data.numpy().reshape((-1, self.net.n_tasks)))
      if len(inputs[0]) == 0 or (inputs[0][0].shape[0] == X.shape[0] and len(inputs[0]) < 64):
        inputs[0].append(X)
        inputs[1].append(X_metas)
      elif inputs[0][0].shape[0] != X.shape[0] or len(inputs[0]) == 64:
        input_X = t.cat(inputs[0], 1)
        input_X_metas = t.cat(inputs[1], 0)
        if self.opt.gpu:
          input_X = input_X.cuda()
          input_X_metas = input_X_metas.cuda()
        output = self.net.predict(input_X, input_X_metas, len(inputs[0]))
        test_preds.append(output)
        inputs[0] = [X]
        inputs[1] = [X_metas]
      
    input_X = t.cat(inputs[0], 1)
    input_X_metas = t.cat(inputs[1], 0)
    if self.opt.gpu:
      input_X = input_X.cuda()
      input_X_metas = input_X_metas.cuda()
    output = self.net.predict(input_X, input_X_metas, len(inputs[0]))
    test_preds.append(output)
    
    test_preds_output = []
    for p in test_preds:
      test_preds_output.extend([s for s in p])
    assert len(test_preds_output) == n_points
    assert len(labels) == n_points
    return labels, test_preds_output
  
  def evaluate(self, test_data_mapping):
    labels, test_preds = self.predict(test_data_mapping)
    cos_sims = []
    for label, pred in zip(labels, test_preds):
      sim = np.sum(label * pred)/np.sqrt(np.sum(label*label) * np.sum(pred*pred))
      cos_sims.append(sim)
    print("Cos similarity %f" % np.mean(cos_sims))
    return cos_sims


  @staticmethod
  def load_data_from_files(test_data_mapping):
    test_data = {}
    files_included = set([test_data_mapping[i][0] for i in test_data_mapping])
    dat = {}
    for f in files_included:
      dat.update(np.load(f).item())
    for i in test_data_mapping:
      test_data[i] = dat[i]
    return test_data


class SpectraDataset(Dataset):
  def __init__(self, sample_file_mapping, batch_size, train=True):
    """ train: set to True to shuffle the order and speed up training
    """
    self.sample_IDs = list(sample_file_mapping.keys())
    self.n_samples = len(self.sample_IDs)
    self.n_batches = int(np.ceil(float(self.n_samples) / batch_size))
    
    self.sample_file_mapping = sample_file_mapping    
    self.batch_size = batch_size

    self.cached = set()
    self.dat = {}

    if type(self.sample_file_mapping[self.sample_IDs[0]][0]) is str:
      if train:
        self.sample_IDs = sorted(self.sample_IDs, key=lambda x: (sample_file_mapping[x][1], sample_file_mapping[x][0]))
      self.batch_samples_IDs = [self.sample_IDs[index*self.batch_size:(index+1)*self.batch_size] \
                                for index in range(self.n_batches)]
      self.files_included = [list(set([self.sample_file_mapping[i][0] for i in batch_samples])) \
                             for batch_samples in self.batch_samples_IDs]
      if train:
        np.random.seed(123)
        all_files = set()
        for fs in self.files_included:
          all_files |= set(fs)
        all_files = sorted(list(all_files))
        np.random.shuffle(all_files)
        order = sorted(np.arange(len(self.files_included)), key=lambda x: all_files.index(self.files_included[x][0]))
        self.batch_samples_IDs = [self.batch_samples_IDs[i] for i in order]
        self.files_included = [self.files_included[i] for i in order]
    else:
      if train:
        self.sample_IDs = sorted(self.sample_IDs, key=lambda x: sample_file_mapping[x][0].shape[0])
      self.batch_samples_IDs = [self.sample_IDs[index*self.batch_size:(index+1)*self.batch_size] \
                                for index in range(self.n_batches)]
      if train:
        np.random.seed(123)
        np.random.shuffle(self.batch_samples_IDs)
  
  def __len__(self):
    return self.n_batches

  def __getitem__(self, index):
    
    batch_samples = self.batch_samples_IDs[index]
    
    if type(self.sample_file_mapping[batch_samples[0]][0]) is str:
      files_included = set(self.files_included[index])
      
      for f in files_included - self.cached:
        self.dat.update(np.load(f).item())
        self.cached.add(f)
      
      batch_raw = [self.dat[i] for i in batch_samples]
      if len(self.cached) > 5:
        self.dat.clear()
        self.cached.clear()
    else:
      batch_raw = [self.sample_file_mapping[i] for i in batch_samples]
    batch = self.assemble_batch(batch_raw)

    X = Variable(t.from_numpy(batch[0])).float()
    X_metas = Variable(t.from_numpy(batch[1])).float()
    y = Variable(t.from_numpy(batch[2])).float()
    weights = Variable(t.from_numpy(batch[3])).float()
    return (X, X_metas, y, weights)
  
  @staticmethod
  def sample_weight(sample):
    return 1.
  
  def assemble_batch(self, batch_raw):
    # Assemble samples
    batch_length = np.max([sample[0].shape[0] for sample in batch_raw])
    out_batch_X = []
    out_batch_X_metas = []
    out_batch_y = []
    batch_weights = []
    for sample in batch_raw:
      sample_length = sample[0].shape[0]
      out_batch_X.append(np.pad(sample[0], ((0, batch_length - sample_length), (0, 0)), 'constant'))
      out_batch_X_metas.append(sample[1])
      out_batch_y.append(np.pad(sample[2], ((0, batch_length - sample_length), (0, 0), (0, 0)), 'constant'))
      batch_weights.append(self.sample_weight(sample))
      assert out_batch_X[-1].shape[0] - out_batch_y[-1].shape[0] == 1
      
    if len(out_batch_X) < self.batch_size:
      pad_length = self.batch_size - len(out_batch_X)
      out_batch_X.extend([out_batch_X[0]] * pad_length)
      out_batch_X_metas.extend([out_batch_X_metas[0]] * pad_length)
      out_batch_y.extend([out_batch_y[0]] * pad_length)
      batch_weights.extend([0.] * pad_length)
      
    batch = (np.stack(out_batch_X, axis=1), 
             np.stack(out_batch_X_metas, axis=0), 
             np.stack(out_batch_y, axis=1),
             np.array(batch_weights).reshape((-1,)))
    return batch
