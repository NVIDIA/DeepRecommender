# Copyright (c) 2017 NVIDIA Corporation
import unittest
from reco_encoder.data.input_layer import UserItemRecDataProvider

class UserItemRecDataProviderTest(unittest.TestCase):
  def test_1(self):
    print("Test 1 started")
    params = {}
    params['batch_size'] = 64
    params['data_dir'] = 'test/testData_iRec'
    data_layer = UserItemRecDataProvider(params=params)
    print("Total items found: {}".format(len(data_layer.data.keys())))
    self.assertTrue(len(data_layer.data.keys())>0)

  def test_iterations(self):
    params = {}
    params['batch_size'] = 32
    params['data_dir'] = 'test/testData_iRec'
    data_layer = UserItemRecDataProvider(params=params)
    print("Total items found: {}".format(len(data_layer.data.keys())))
    for i, data in enumerate(data_layer.iterate_one_epoch()):
      print(i)
      print(data.size())

if __name__ == '__main__':
  unittest.main()
