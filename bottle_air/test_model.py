import unittest
from config import config
from model import *
from mockcache import Client
import tensorflow as tf
from db import *
import os
from hyperopt import STATUS_OK
from keras.backend import clear_session

class TestModel(unittest.TestCase):
    def setUp(self):
      config.tf_server = tf.train.Server.create_local_server()
      config.mc = Client()
      self.model = Model()
      self.model.data = {"x": [0, 0.1, 0.2, 0.3], "x2": [5, -2, 3, 10], "output_y": [1, 2, 3, 4]}
      self.model.norms = {"x": [0, 0.3], "x2": [-2, 10], "output_y": [1, 4]}
      self.model.types = {"x": "float", "x2": "float"}
      self.model.model_path = "/tmp/mymodel"
      
      self.model2 = Model()
      self.model2.data = {"x0": [0, 0.1, 0.2, 0.3], "class": ["a", "a", "c", "k"], "output_y": [1, 2, 3, 4]}
      self.model2.norms = {"x0": [0, 0.3], "class": [], "output_y": [1, 4]}
      self.model2.types = {"x0": "float", "class": "str"}
      self.model2.model_path = "/tmp/mymodel2"
      
    def deleteFile(self, f):
      try:
        os.remove(f)
      except OSError:
        pass
      
      try:
        os.remove(f + "_keras")
      except OSError:
        pass
      
    def tearDown(self):
      self.deleteFile(self.model.model_path)
      self.deleteFile(self.model2.model_path)
        
      clear_session()
    
    def assertEqualsArrays(self, arr1, arr2):
      for idx, _ in enumerate(arr1):
        self.assertEquals(arr1[idx], arr2[idx])

    def test_get_data_sets(self):
      X, Y = self.model.get_data_sets()
      self.assertEquals(len(X), 1)
      for idx, val in enumerate(X[0]):
        self.assertEquals(val[1], self.model.data["x"][idx])
        self.assertEquals(val[0], self.model.data["x2"][idx])
      self.assertEqualsArrays(Y, self.model.data["output_y"])

    def test_run_model(self):
      model_fn = self.model.run_model()
      dummy = []
      layer1 = [dummy, [3, 'relu']]
      layer2 = [dummy, [4, 'linear']]
      layer3 = [dummy, [5, 'sigmoid']]
      layer4 = [dummy, [10, 'tanh']]
      
      layers = [dummy, [layer1, layer2, layer3, layer4]]
      retval = model_fn({"optimizer" : "adagrad", "layers": layers})
      self.assertTrue(retval["loss"] < 4)
      self.assertEquals(retval["status"], STATUS_OK)
      
    def test_run_infer_run(self):
      model_fn = self.model.run_model(persist=True)
      dummy = []
      layer1 = [dummy, [3, 'relu']]
      layer2 = [dummy, [4, 'linear']]
      layer3 = [dummy, [5, 'sigmoid']]
      layer4 = [dummy, [10, 'tanh']]
      
      layers = [dummy, [layer1, layer2, layer3, layer4]]
      retval = model_fn({"optimizer" : "adagrad", "layers": layers})
      self.assertTrue(retval["loss"] < 4)
      self.assertEquals(retval["status"], STATUS_OK)
      
      output = self.model.infer({"x": [1, 10, 100], "x2": [-1, -10, -100]})
      self.assertTrue("best" in output and len(output["best"]) == 3)
      
      
      ## Now try to train a second model...
      clear_session()
      model_fn = self.model2.run_model(persist=True)
      dummy = []
      layer1 = [dummy, [3, 'relu']]
      layer2 = [dummy, [4, 'linear']]
      layer3 = [dummy, [5, 'sigmoid']]
      layer4 = [dummy, [10, 'tanh']]
      
      layers = [dummy, [layer1, layer2, layer3, layer4]]
      print 'lala ' + str(self.model2.get_data_sets())
      retval = model_fn({"optimizer" : "adagrad", "layers": layers})
      self.assertTrue(retval["loss"] < 4)
      self.assertEquals(retval["status"], STATUS_OK)

if __name__ == '__main__':
    unittest.main()
