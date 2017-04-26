import json
from keras.models import Sequential
from keras.layers import Dense, Activation

import numpy as np
import os.path

WIDE_RANGE = 5
DEEP_RANGE = 5

class ModelStatus():
  NULL = 0
  CREATED = 1
    
class Model():
  model_path = ''
  train_files = []
  status = ModelStatus.NULL
  types = {}
  data = {}
  keras_models = []

  def __init__(self, path=''):
    self.model_path = path
    self.status = ModelStatus.CREATED
    
  def get_handle(self):
    return self.model_path.split('/')[-1]
    
  def add_train_file(self, filename):
    from db import load_csvs
    self.train_files.append(filename)
    data, types = load_csvs(self.train_files)
    if not self.data:
      self.data = data
      self.types = types
    else:
      for idx, _ in enumerate(types):
        if types[idx] != self.types[idx]:
          return 'Error: Mismatching types with existing model %s, %s' % (str(types), str(self.types))
      for k in self.data.iterkeys():
        self.data[k].extend(data[k])

  def start_training(self):
    start_depth = 1
    start_width = len(types)
    
    

  def from_json(self, json_str):
    json_obj = json.loads(json_str)
    members = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self,a))]
    for member in members:
      setattr(self, member, json_obj[member])
    
  def to_json(self):
    members = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self,a))]
    return json.dumps({member: getattr(self, member) for member in members})
