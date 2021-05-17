# Copyright (c) 2017 NVIDIA Corporation

# parameters to run benchmark on cpu
# --path_to_train_data Netflix/N1W_TRAIN --path_to_eval_data Netflix/N1W_TEST --hidden_layers 512,512,1024 --non_linearity_type selu --save_path model_save/model.epoch_0 --drop_prob 0.8 --predictions_path preds.txt --nooutput --forcecpu

# parameters to run benchmark on cuda
# --path_to_train_data Netflix/N1W_TRAIN --path_to_eval_data Netflix/N1W_TEST --hidden_layers 512,512,1024 --non_linearity_type selu --save_path model_save/model.epoch_0 --drop_prob 0.8 --predictions_path preds.txt --nooutput --forcecuda

import torch
import argparse
import copy
import time
from reco_encoder.data import input_layer
from reco_encoder.model import model
from torch.autograd import Variable
from pathlib import Path
import torch.autograd.profiler as profiler

def getCommandLineArgs() :
  parser = argparse.ArgumentParser(description='RecoEncoder')
  
  parser.add_argument('--drop_prob', type=float, default=0.0, metavar='N',
                      help='dropout drop probability')
  parser.add_argument('--constrained', action='store_true',
                      help='constrained autoencoder')
  parser.add_argument('--skip_last_layer_nl', action='store_true',
                      help='if present, decoder\'s last layer will not apply non-linearity function')
  parser.add_argument('--hidden_layers', type=str, default="1024,512,512,128", metavar='N',
                      help='hidden layer sizes, comma-separated')
  parser.add_argument('--path_to_train_data', type=str, default="", metavar='N',
                      help='Path to training data')
  parser.add_argument('--path_to_eval_data', type=str, default="", metavar='N',
                      help='Path to evaluation data')
  parser.add_argument('--non_linearity_type', type=str, default="selu", metavar='N',
                      help='type of the non-linearity used in activations')
  parser.add_argument('--save_path', type=str, default="autorec.pt", metavar='N',
                      help='where to save model')
  parser.add_argument('--predictions_path', type=str, default="out.txt", metavar='N',
                      help='where to save predictions')
  parser.add_argument('--jit', action='store_true',
                      help='jit-ify model before running')
  parser.add_argument('--forcecuda', action='store_true',
                      help='force cuda use')
  parser.add_argument('--forcecpu', action='store_true',
                      help='force cpu use')
  parser.add_argument('--nooutput', action='store_true',
                      help='disable writing output to file')
  parser.add_argument('--profile', action='store_true',
                      help='enable profiler and stat print')
  
  args = parser.parse_args()

  return args

def getBenchmarkArgs(forceCuda):

  class Args:
    pass
  
  args = Args()

  args.drop_prob          = 0.8
  args.constrained        = False
  args.skip_last_layer_nl = False
  args.hidden_layers      = '512,512,1024'
  args.path_to_train_data = 'Netflix/N1W_TRAIN'
  args.path_to_eval_data  = 'Netflix/N1W_TEST'
  args.non_linearity_type = 'selu'
  args.save_path          = 'model_save/model.epoch_0'
  args.predictions_path   = 'preds.txt'
  args.jit                = False
  args.forcecuda          = forceCuda
  args.forcecpu           = not forceCuda
  args.nooutput           = True
  args.profile            = False

  return args

def processArgState(args) :

  print(args)
  
  if args.forcecpu and args.forcecuda:
      print("Error, force cpu and cuda cannot both be set")
      quit()
  
  args.use_cuda = torch.cuda.is_available() # global flag
  if args.use_cuda:
      print('GPU is available.') 
  else: 
      print('GPU is not available.')
  
  if args.use_cuda and args.forcecpu:
      args.use_cuda = False
  
  if args.use_cuda:
      print('Running On GPU')
  else:
      print('Running On CUDA')
  
  if args.profile:
      print('Profiler Enabled')

  return args

class DeepRecommenderInferenceBenchmark:

  def __init__(self, device = 'cpu', jit=False, usecommandlineargs = False) :

    if device == "cpu":
      forcecuda = False
    elif device == "cuda":
      forcecuda = True
    else:
      raise Exception ("Error unsupported device type")
    
    # jit not supported, error here if jit is requested
    if jit == True:
      raise Exception ("Jit Mode Not Supported")

    if usecommandlineargs:
      self.args = getCommandLineArgs()
    else:
      self.args = getBenchmarkArgs(forcecuda)

    args = processArgState(self.args)

    self.params = dict()
    self.params['batch_size'] = 1
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
    self.eval_params['batch_size'] = 1
    self.eval_params['data_dir'] = self.args.path_to_eval_data
    self.eval_data_layer = input_layer.UserItemRecDataProvider(params=self.eval_params,
                                                               user_id_map=self.data_layer.userIdMap,
                                                               item_id_map=self.data_layer.itemIdMap)
  
    self.rencoder = model.AutoEncoder(layer_sizes=[self.data_layer.vector_dim] + [int(l) for l in self.args.hidden_layers.split(',')],
                                      nl_type=self.args.non_linearity_type,
                                      is_constrained=self.args.constrained,
                                      dp_drop_prob=self.args.drop_prob,
                                      last_layer_activations=not self.args.skip_last_layer_nl)
  
    self.path_to_model = Path(self.args.save_path)
    if self.path_to_model.is_file():
      print("Loading model from: {}".format(self.path_to_model))
      self.rencoder.load_state_dict(torch.load(self.args.save_path))
  
    if not self.args.nooutput:
      print('######################################################')
      print('######################################################')
      print('############# AutoEncoder Model: #####################')
      print(self.rencoder)
      print('######################################################')
      print('######################################################')

    self.rencoder.eval()
  
    if self.args.use_cuda: self.rencoder = self.rencoder.cuda()
  
    if self.args.jit:
      self.rencoder = torch.jit.script(self.rencoder)
  
    self.inv_userIdMap = {v: k for k, v in self.data_layer.userIdMap.items()}
    self.inv_itemIdMap = {v: k for k, v in self.data_layer.itemIdMap.items()}
  
    self.eval_data_layer.src_data = self.data_layer.data

  def eval(self, niter=1):
      for i, ((out, src), majorInd) in enumerate(self.eval_data_layer.iterate_one_epoch_eval(for_inf=True)):
        inputs = Variable(src.cuda().to_dense() if self.args.use_cuda else src.to_dense())
        targets_np = out.to_dense().numpy()[0, :]
  
        outputs = self.rencoder(inputs).cpu().data.numpy()[0, :]
        non_zeros = targets_np.nonzero()[0].tolist()
        major_key = self.inv_userIdMap [majorInd]
  
        if not self.args.nooutput:
            with open(self.args.predictions_path, 'w') as outf:
              for ind in non_zeros:
                outf.write("{}\t{}\t{}\t{}\n".format(major_key, self.inv_itemIdMap[ind], self.outputs[ind], targets_np[ind]))
              if i % 10000 == 0:
                print("Done: {}".format(i))

  def TimedInferenceRun(self) :
  
      print('Timed Inference Start')
  
      e_start_time = time.time()
  
      if self.args.profile:
        with profiler.profile(record_shapes=True, use_cuda=True) as prof:
          with profiler.record_function("Inference"):
            self.eval()
      else:
        self.eval()
  
      e_end_time = time.time()
  
      print('Timed Inference Complete')
      print('Inference finished in {} seconds'
            .format(e_end_time - e_start_time))
  
      if self.args.profile:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))        
        prof.export_chrome_trace("trace.json")

def main():
  benchmarkCuda = DeepRecommenderInferenceBenchmark(device='cuda')
  benchmarkCuda.TimedInferenceRun()

  benchmarkCPU = DeepRecommenderInferenceBenchmark(device='cpu')
  benchmarkCPU.TimedInferenceRun()

if __name__ == '__main__':
  main()
