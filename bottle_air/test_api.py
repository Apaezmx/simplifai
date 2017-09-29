import unittest
from config import config
from api import *
import tensorflow as tf
from db import *
import os
import os.path
import shutil
from hyperopt import STATUS_OK
from keras.backend.tensorflow_backend import clear_session

class TestApi(unittest.TestCase):
    def setUp(self):
      self.files_to_delete = []
      if not os.path.isfile("/tmp/x.csv"):
        shutil.copy2(os.path.dirname(os.path.realpath(__file__)) +"/test_files/x.csv", "/tmp")
     
      config.tf_server = tf.train.Server.create_local_server()
      config.ROOT_PATH = "/tmp"
      
    def tearDown(self):
      self.deleteFile("/tmp/x.csv")
      config.ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
      
      for f in self. files_to_delete:
        self.deleteFile(f)
      
    def deleteFile(self, f):
      try:
        os.remove(f)
      except OSError:
        pass
    
    def assertEqualsArrays(self, arr1, arr2):
      for idx, _ in enumerate(arr1):
        self.assertEquals(arr1[idx], arr2[idx])

    def test_upload_csv(self):
      pass  # TODO
      #response = api.upload_csv({}, {"upload": "x.csv"})
      #self.assertTrue(isinstance(response, dict))
      #handle = response["handle"]
      #self.files_to_delete.append("/tmp/" + handle)
      #self.assertTrue(response["status"], "OK")
      
      

if __name__ == '__main__':
    unittest.main()
