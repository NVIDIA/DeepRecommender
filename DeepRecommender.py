import run
from run import DeepRecommenderTrainBenchmark
import infer
from infer import DeepRecommenderInferenceBenchmark

class DeepRecommenderBenchmark:
  def __init__(self, device="cpu", jit=False):
    self.train = DeepRecommenderTrainBenchmark(device = device, jit = jit)
    self.infer = DeepRecommenderInferenceBenchmark(device = device, jit = jit)

  def train(self, niter=1):
    self.train.train(train.args.num_epochs)
  
  def eval(self, niter=1):
    self.infer.eval(self, niter)

  def timedInfer(self):
    self.infer.TimedInferenceRun()

  def timedTrain(self):
    self.train.TimedTrainingRun()

def main():
  cudaBenchMark = DeepRecommenderBenchmark(device = 'cuda', jit = False)
  cudaBenchMark.timedTrain()
  cudaBenchMark.timedInfer()

  cpuBenchMark = DeepRecommenderBenchmark(device = 'cpu', jit = False)
  cpuBenchMark.timedTrain()
  cpuBenchMark.timedInfer()

if __name__ == '__main__':
  main()
