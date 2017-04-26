import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing.sequence import pad_sequences
from keras_utils import *

import numpy as np
import os.path

WIDE_RANGE = 5
DEEP_RANGE = 5
WIDE_RATIO = 5  # For every input, we will put WIDE_RATIO neurons in layer 1.

class ModelStatus():
  NULL = 0
  CREATED = 1
    
class Model():
  model_path = ''
  train_files = []
  status = ModelStatus.NULL
  types = {}
  data = {}
  embedding_dicts = {}
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
    output_headers = [outputs for outputs in self.data.iterkeys() if outputs.startsWith('output_')]
    if not output_headers:
      raise ValueError('No outputs defined!')
    
    # Process string features.
    string_features = {}
    for header, typ in self.types.iteritems():
      if typ != 'str'
        continue
      # Every string feature is treated as a list of words.
      word_list = [x.split() for x in self.data[header]]
      dict_, _ = process_text_feature(word_list)
      self.embedding_dicts[header] = dict_
      lengths = [len(words) for words in word_list]
      lengths.sort()
      input_size = lengths[int(len(lengths) * 0.9)]
      for idx, words in enumerate(word_list):
        # Strings to integers. Pad sequences with zeros so that all of them have the same size.
        word_list[idx] = pad_sequences([[dict_[word] for word in word_list]], maxlen=input_size, padding='post', truncating='post')[0]
      string_features[header] = word_list
    
    # Build models.
    # Merge all inputs into one model.
    def init_model(self, string_features=string_features):
      feature_models = []
      total_input_size = 0
      for header, word_list in string_features.iteriterms():
        embedding_size = np.round(np.log10(len(self.embedding_dicts[header])))
        model = Sequential()
        model.add(Embedding(len(self.embedding_dicts[header].keys()), embedding_size, input_length=1))
        model.add(Flatten())
        total_input_size += embedding_size * len(word_list[0])
        feature_models.append(model)
      
      numeric_inputs = len(self.data) - len(string_features)
      num_model = Sequential()
      num_model.add(Dense(numeric_inputs, input_shape=(numeric_inputs,)))
      num_model.add(Activation('relu'))
      total_input_size += numeric_inputs
      feature_models.append(num_model)
      
      merged_model = Sequential()
      merged_model.add(Merge(feature_models, mode='concat', concat_axis=1))
      return merged_model, total_input_size
    
    for depth in range(start_depth, start_depth + DEEP_RANGE):
      for width in range(start_width, start_width + WIDE_RANGE):
        model, input_size = init_model()
        model.add(Dense(2048, input_shape=(input_size,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
      
      
    

  def from_json(self, json_str):
    json_obj = json.loads(json_str)
    members = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self,a))]
    for member in members:
      setattr(self, member, json_obj[member])
    
  def to_json(self):
    members = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self,a))]
    return json.dumps({member: getattr(self, member) for member in members})
