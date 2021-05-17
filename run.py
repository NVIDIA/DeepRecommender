# Copyright (c) 2017 NVIDIA Corporation

# to run against cuda:
# --gpu_ids 0 --path_to_train_data Netflix/N1W_TRAIN --path_to_eval_data Netflix/N1W_VALID --hidden_layers 512,512,1024 --non_linearity_type selu --batch_size 128 --logdir model_save --drop_prob 0.8 --optimizer momentum --lr 0.005 --weight_decay 0 --aug_step 1 --noise_prob 0 --num_epochs 1 --summary_frequency 1000 --forcecuda

# to run on cpu:
# --gpu_ids 0 --path_to_train_data Netflix/N1W_TRAIN --path_to_eval_data Netflix/N1W_VALID --hidden_layers 512,512,1024 --non_linearity_type selu --batch_size 128 --logdir model_save --drop_prob 0.8 --optimizer momentum --lr 0.005 --weight_decay 0 --aug_step 1 --noise_prob 0 --num_epochs 1 --summary_frequency 1000 --forcecpu


import torch
import argparse
from reco_encoder.data import input_layer
from reco_encoder.model import model
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
from torch.autograd import Variable
import copy
import time
from pathlib import Path
from logger import Logger
from math import sqrt
import numpy as np
import os
import torch.autograd.profiler as profiler

def getTrainBenchmarkArgs() :

  class Args:
    pass

  args = Args()
  args.lr = 0.005
  args.weight_decay = 0
  args.drop_prob = 0.8
  args.noise_prob = 0
  args.batch_size = 128
  args.summary_frequency  = 1000
  args.aug_step           = 1
  args.constrained        = False
  args.skip_last_layer_nl = False
  args.num_epochs         = 1
  args.save_every         = 3
  args.optimizer          = 'momentum'
  args.hidden_layers      = '512,512,1024'
  args.gpu_ids            = '0'
  args.path_to_train_data = 'Netflix/N1W_TRAIN'
  args.path_to_eval_data  = 'Netflix/N1W_VALID'
  args.non_linearity_type = 'selu'
  args.logdir             = 'model_save'
  args.nooutput           = True
  args.forcecuda          = False
  args.forcecpu           = False
  args.profile            = False

  return args

def getTrainCommandLineArgs() :

  parser = argparse.ArgumentParser(description='RecoEncoder')
  parser.add_argument('--lr', type=float, default=0.00001, metavar='N',
                      help='learning rate')
  parser.add_argument('--weight_decay', type=float, default=0.0, metavar='N',
                      help='L2 weight decay')
  parser.add_argument('--drop_prob', type=float, default=0.0, metavar='N',
                      help='dropout drop probability')
  parser.add_argument('--noise_prob', type=float, default=0.0, metavar='N',
                      help='noise probability')
  parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                      help='global batch size')
  parser.add_argument('--summary_frequency', type=int, default=100, metavar='N',
                      help='how often to save summaries')
  parser.add_argument('--aug_step', type=int, default=-1, metavar='N',
                      help='do data augmentation every X step')
  parser.add_argument('--constrained', action='store_true',
                      help='constrained autoencoder')
  parser.add_argument('--skip_last_layer_nl', action='store_true',
                      help='if present, decoder\'s last layer will not apply non-linearity function')
  parser.add_argument('--num_epochs', type=int, default=50, metavar='N',
                      help='maximum number of epochs')
  parser.add_argument('--save_every', type=int, default=3, metavar='N',
                      help='save every N number of epochs')
  parser.add_argument('--optimizer', type=str, default="momentum", metavar='N',
                      help='optimizer kind: adam, momentum, adagrad or rmsprop')
  parser.add_argument('--hidden_layers', type=str, default="1024,512,512,128", metavar='N',
                      help='hidden layer sizes, comma-separated')
  parser.add_argument('--gpu_ids', type=str, default="0", metavar='N',
                      help='comma-separated gpu ids to use for data parallel training')
  parser.add_argument('--path_to_train_data', type=str, default="", metavar='N',
                      help='Path to training data')
  parser.add_argument('--path_to_eval_data', type=str, default="", metavar='N',
                      help='Path to evaluation data')
  parser.add_argument('--non_linearity_type', type=str, default="selu", metavar='N',
                      help='type of the non-linearity used in activations')
  parser.add_argument('--logdir', type=str, default="logs", metavar='N',
                      help='where to save model and write logs')
  parser.add_argument('--nooutput', action='store_true',
                      help='disable writing output to file')
  parser.add_argument('--forcecuda', action='store_true',
                      help='force cuda use')
  parser.add_argument('--forcecpu', action='store_true',
                      help='force cpu use')
  parser.add_argument('--profile', action='store_true',
                      help='enable profiler and stat print')
  
  args = parser.parse_args()

  return args

def processTrainArgState(args) :

  print(args)
  
  if args.forcecpu and args.forcecuda:
    print("Error, force cpu and cuda cannot bother be set")
    quit()

  args.use_cuda = torch.cuda.is_available() # global flag
  if args.use_cuda:
    print('GPU is available.') 
  else: 
    print('GPU is not available.')
  
  if args.use_cuda and args.forcecpu:
    args.use_cuda = False
  
  if args.use_cuda:
    print('Running On CUDA')
  else:
    print('Running On CPU')

  return args

def log_var_and_grad_summaries(logger, layers, global_step, prefix, log_histograms=False):
  """
  Logs variable and grad stats for layer. Transfers data from GPU to CPU automatically
  :param logger: TB logger
  :param layers: param list
  :param global_step: global step for TB
  :param prefix: name prefix
  :param log_histograms: (default: False) whether or not log histograms
  :return:
  """
  for ind, w in enumerate(layers):
    # Variables
    w_var = w.data.cpu().numpy()
    logger.scalar_summary("Variables/FrobNorm/{}_{}".format(prefix, ind), np.linalg.norm(w_var),
                          global_step)
    if log_histograms:
      logger.histo_summary(tag="Variables/{}_{}".format(prefix, ind), values=w.data.cpu().numpy(),
                           step=global_step)

    # Gradients
    w_grad = w.grad.data.cpu().numpy()
    logger.scalar_summary("Gradients/FrobNorm/{}_{}".format(prefix, ind), np.linalg.norm(w_grad),
                          global_step)
    if log_histograms:
      logger.histo_summary(tag="Gradients/{}_{}".format(prefix, ind), values=w.grad.data.cpu().numpy(),
                         step=global_step)

def DoTrainEval(encoder, evaluation_data_layer, use_cuda):
  encoder.eval()
  denom = 0.0
  total_epoch_loss = 0.0
  for i, (eval, src) in enumerate(evaluation_data_layer.iterate_one_epoch_eval()):
    inputs = Variable(src.cuda().to_dense() if use_cuda else src.to_dense())
    targets = Variable(eval.cuda().to_dense() if use_cuda else eval.to_dense())
    outputs = encoder(inputs)
    loss, num_ratings = model.MSEloss(outputs, targets)
    total_epoch_loss += loss.item()
    denom += num_ratings.item()
  return sqrt(total_epoch_loss / denom)

class DeepRecommenderTrainBenchmark:

  def __init__(self, device="cpu", jit=False, processCommandLine = False):
    self.TrainInit(device, jit, processCommandLine)


  def TrainInit(self, device="cpu", jit=False, processCommandLine = False):

    if (processCommandLine) :
      self.args = getTrainCommandLineArgs()
    else:
      self.args = getTrainBenchmarkArgs()

      if device == "cpu":
        forcecuda = False
      elif device == "cuda":
        forcecuda = True
      else:
        raise Exception ("Error unsupported device type")

      self.args.forcecuda = forcecuda
      self.args.forcecpu = not forcecuda

      # jit not supported, error here if jit is requested
      if jit == True:
        raise Exception ("Jit Mode Not Supported")

    self.args = processTrainArgState(self.args)

    self.logger = Logger(self.args.logdir)
    self.params = dict()
    self.params['batch_size'] = self.args.batch_size
    self.params['data_dir'] =  self.args.path_to_train_data
    self.params['major'] = 'users'
    self.params['itemIdInd'] = 1
    self.params['userIdInd'] = 0
    print("Loading training data")
    self.data_layer = input_layer.UserItemRecDataProvider(params=self.params)
    print("Data loaded")
    print("Total items found: {}".format(len(self.data_layer.data.keys())))
    print("Vector dim: {}".format(self.data_layer.vector_dim))
  
    print("Loading eval data")
    self.eval_params = copy.deepcopy(self.params)
    # must set eval batch size to 1 to make sure no examples are missed
    self.eval_params['data_dir'] = self.args.path_to_eval_data
    self.eval_data_layer = input_layer.UserItemRecDataProvider(params=self.eval_params,
                                                               user_id_map=self.data_layer.userIdMap, # the mappings are provided
                                                               item_id_map=self.data_layer.itemIdMap)
    self.eval_data_layer.src_data = self.data_layer.data
    self.rencoder = model.AutoEncoder(layer_sizes=[self.data_layer.vector_dim] + [int(l) for l in self.args.hidden_layers.split(',')],
                                      nl_type=self.args.non_linearity_type,
                                      is_constrained=self.args.constrained,
                                      dp_drop_prob=self.args.drop_prob,
                                      last_layer_activations=not self.args.skip_last_layer_nl)
    os.makedirs(self.args.logdir, exist_ok=True)
    self.model_checkpoint = self.args.logdir + "/model"
    self.path_to_model = Path(self.model_checkpoint)
    if self.path_to_model.is_file():
      print("Loading model from: {}".format(self.model_checkpoint))
      self.rencoder.load_state_dict(torch.load(self.model_checkpoint))
  
    if not self.args.nooutput:
      print('######################################################')
      print('######################################################')
      print('############# AutoEncoder Model: #####################')
      print(self.rencoder)
      print('######################################################')
      print('######################################################')
  
    if self.args.use_cuda:
      gpu_ids = [int(g) for g in self.args.gpu_ids.split(',')]
      print('Using GPUs: {}'.format(gpu_ids))
      if len(gpu_ids)>1:
        self.rencoder = nn.DataParallel(self.rencoder,
                                   device_ids=gpu_ids)
      self.rencoder = self.rencoder.cuda()
  
    if self.args.optimizer == "adam":
      self.optimizer = optim.Adam(self.rencoder.parameters(),
                                  lr=self.args.lr,
                                  weight_decay=self.args.weight_decay)
    elif self.args.optimizer == "adagrad":
      self.optimizer = optim.Adagrad(self.rencoder.parameters(),
                                lr=self.args.lr,
                                weight_decay=self.args.weight_decay)
    elif self.args.optimizer == "momentum":
      self.optimizer = optim.SGD(self.rencoder.parameters(),
                            lr=self.args.lr, momentum=0.9,
                            weight_decay=self.args.weight_decay)
      self.scheduler = MultiStepLR(self.optimizer, milestones=[24, 36, 48, 66, 72], gamma=0.5)
    elif args.optimizer == "rmsprop":
      self.optimizer = optim.RMSprop(self.rencoder.parameters(),
                                lr=self.args.lr, momentum=0.9,
                                weight_decay=self.args.weight_decay)
    else:
      raise  ValueError('Unknown optimizer kind')
  
    self.t_loss = 0.0
    self.t_loss_denom = 0.0
    self.denom = 0.0
    self.total_epoch_loss = 0.0
    self.global_step = 0
  
    if self.args.noise_prob > 0.0:
      self.dp = nn.Dropout(p=self.args.noise_prob)

  def DoTrain(self):
  
    self.rencoder.train()
    #if self.args.optimizer == "momentum":
    #  self.scheduler.step()
  
    for i, mb in enumerate(self.data_layer.iterate_one_epoch()):
  
      inputs = Variable(mb.cuda().to_dense() if self.args.use_cuda else mb.to_dense())
      self.optimizer.zero_grad()
  
      outputs = self.rencoder(inputs)
  
      loss, num_ratings = model.MSEloss(outputs, inputs)
      loss = loss / num_ratings
      loss.backward()
      self.optimizer.step()
      self.global_step += 1
      self.t_loss += loss.item()
      self.t_loss_denom += 1
  
      if not self.args.nooutput:
        if i % self.args.summary_frequency == 0:
          print('[%d, %5d] RMSE: %.7f' % (self.epoch, i, sqrt(self.t_loss / self.t_loss_denom)))
          self.logger.scalar_summary("Training_RMSE", sqrt(self.t_loss/self.t_loss_denom), self.global_step)
          self.t_loss = 0
          self.t_loss_denom = 0.0
          log_var_and_grad_summaries(self.logger, self.rencoder.encode_w, self.global_step, "Encode_W")
          log_var_and_grad_summaries(self.logger, self.rencoder.encode_b, self.global_step, "Encode_b")
          if not self.rencoder.is_constrained:
            log_var_and_grad_summaries(self.logger, self.rencoder.decode_w, self.global_step, "Decode_W")
          log_var_and_grad_summaries(self.logger, self.rencoder.decode_b, self.global_step, "Decode_b")
  
      self.total_epoch_loss += loss.item()
      self.denom += 1
  
      #if args.aug_step > 0 and i % args.aug_step == 0 and i > 0:
      if self.args.aug_step > 0:
        # Magic data augmentation trick happen here
        for t in range(self.args.aug_step):
          inputs = Variable(outputs.data)
          if self.args.noise_prob > 0.0:
            inputs = dp(inputs)
          self.optimizer.zero_grad()
          outputs = self.rencoder(inputs)
          loss, num_ratings = model.MSEloss(outputs, inputs)
          loss = loss / num_ratings
          loss.backward()
          self.optimizer.step()
  
  def train(self, niter=1) :
    for self.epoch in range(niter):
      print('Doing epoch {} of {}'.format(self.epoch, niter))
      print('Timing Start')
      e_start_time = time.time()
  
      self.DoTrain()
  
      e_end_time = time.time()
      print('Timing End')
  
      if self.args.profile:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))        
        prof.export_chrome_trace("trace.json")
  
      print('Total epoch {} finished in {} seconds with TRAINING RMSE loss: {}'
            .format(self.epoch, e_end_time - e_start_time, sqrt(self.total_epoch_loss/self.denom)))

      if not self.args.nooutput:
        self.logger.scalar_summary("Training_RMSE_per_epoch", sqrt(self.total_epoch_loss/self.denom), self.epoch)
        self.logger.scalar_summary("Epoch_time", e_end_time - e_start_time, self.epoch)
        if self.epoch % self.args.save_every == 0 or self.epoch == self.args.num_epochs - 1:
          eval_loss = DoTrainEval(self.rencoder, self.eval_data_layer, self.args.use_cuda)
          print('Epoch {} EVALUATION LOSS: {}'.format(self.epoch, eval_loss))
  
          self.logger.scalar_summary("EVALUATION_RMSE", eval_loss, self.epoch) 
          print("Saving model to {}".format(self.model_checkpoint + ".epoch_"+str(self.epoch)))
          torch.save(self.rencoder.state_dict(), self.model_checkpoint + ".epoch_"+str(self.epoch))
  
    if not self.args.nooutput:
      print("Saving model to {}".format(self.model_checkpoint + ".last"))
      torch.save(self.rencoder.state_dict(), self.model_checkpoint + ".last")
  
      # save to onnx
      dummy_input = Variable(torch.randn(self.params['batch_size'], self.data_layer.vector_dim).type(torch.float))
      torch.onnx.export(self.rencoder.float(), dummy_input.cuda() if self.args.use_cuda else dummy_input, 
                        self.model_checkpoint + ".onnx", verbose=True)
      print("ONNX model saved to {}!".format(self.model_checkpoint + ".onnx"))

  def TimedTrainingRun(self):
      if self.args.profile:
        with profiler.profile(record_shapes=True, use_cuda=self.args.use_cuda) as prof:
          with profiler.record_function("training_epoch"):
            self.train(self.args.num_epochs)
      else:
        self.train(self.args.num_epochs)

def main() :

  gpuTrain  = DeepRecommenderTrainBenchmark(device = 'cuda')
  gpuTrain.TimedTrainingRun()

  gpuTrain  = DeepRecommenderBenchmark(device = 'cpu')
  gpuTrain.TimedTrainingRun()


if __name__ == '__main__':

  main()
