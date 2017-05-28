import json
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Embedding, Dense, Activation, Merge, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import *
from keras.preprocessing.sequence import pad_sequences
from multiprocessing import Process
from train import train

import numpy as np
import os
import os.path

class ModelStatus():
  NULL = 0
  CREATED = 1
  TRAINING = 2
  TRAINED = 3
    
class Model():
  # Complete path where this Model is saved. It mainly contains metadata. Keras models are store elsewhere.
  model_path = ''
  
  # All files which contain the train data for this Model.
  train_files = []
  
  # Model status on the pipeline.
  status = ModelStatus.NULL
  
  # FeatureName keyed dict which contains strings with the types for each feature.
  types = {}
  
  # FeatureName keyed dict which contains lists with all features.
  data = {}
  
  # Normalization max and min values for numeric inputs.
  norms = {}
  
  # List of tuples containing (string_header, string_column).
  string_features = []
  
  # FeatureName keyed dict for string feature containing the dictionary word->count.
  embedding_dicts = {}
  
  # Keras models history.
  val_losses = {}
  
  # Best model params.
  best_model = ''

  def __init__(self, path=''):
    self.model_path = path
    self.status = ModelStatus.CREATED
    
  def get_handle(self):
    return self.model_path.split('/')[-1]
    
  def add_train_file(self, filename):
    from db import load_csvs
    self.train_files.append(filename)
    data, types, norms = load_csvs(self.train_files)
    if not types:
      return data
    if not self.data:
      self.data = data
      self.types = types
      self.norms = norms
    else:
      for idx, _ in enumerate(types):
        if types[idx] != self.types[idx]:
          return 'Error: Mismatching types with existing model %s, %s' % (str(types), str(self.types))
      for k in self.data.iterkeys():
        self.data[k].extend(data[k])
  
  def process_text_feature(self, feature):
    word_freq = {}
    for f in feature:
      if isinstance(f, list):
        for word in f:
          word_freq[word] = 1 if word not in word_freq else word_freq[word] + 1
      else:
        word_freq[f] = 1 if f not in word_freq else word_freq[f] + 1
    words = {word: i for i, word in enumerate(sorted(word_freq, key=lambda i: -int(word_freq[i])))}
    counts = [word_freq[word] for word in words.keys()]
    return words, counts

  def run_model(self, persist=False):
    def run_model_fn(hp):
      """
        hp: hyperparameter dictionary.
      """
      print str(hp)
      output_headers = [outputs for outputs in self.data.iterkeys() if outputs.startswith('output_')]
      if not output_headers:
        raise ValueError('No outputs defined!')
      
      # Process string features.
      self.string_features = []
      for header, typ in self.types.iteritems():
        if typ != 'str':
          continue
        # Every string feature is treated as a list of words.
        word_list = [x.split() for x in self.data[header]]
        dict_, _ = self.process_text_feature(word_list)
        assert len(dict_) > 0, 'Dict is empty.'
        self.embedding_dicts[header] = dict_
        lengths = [len(words) for words in word_list]
        lengths.sort()
        input_size = lengths[int(np.round(len(lengths) * 0.95))]
        assert input_size > 0, 'input_size is 0.'
        for idx, words in enumerate(word_list):
          # Strings to integers. Pad sequences with zeros so that all of them have the same size.
          word_list[idx] = pad_sequences([[dict_[word] for word in words]], 
                                         maxlen=input_size, padding='post', 
                                         truncating='post')[0].tolist()
        self.string_features.append((header, word_list))
      
      # Build models.
      # Merge all inputs into one model.
      def init_model(self):
        feature_models = []
        total_input_size = 0
        for tup in self.string_features:
          header = tup[0]
          word_list = tup[1]
          sequence_length = len(word_list[0])
          embedding_size = int(np.round(np.log10(len(self.embedding_dicts[header]))))
          embedding_size = embedding_size if embedding_size > 0 else 1
          model = Sequential()
          model.add(Embedding(len(self.embedding_dicts[header].keys()), embedding_size, input_length=sequence_length))
          model.add(Flatten())
          total_input_size += embedding_size * len(word_list[0])
          feature_models.append(model)
        
        numeric_inputs = len(self.data) - len(self.string_features) - len(output_headers)
        num_model = Sequential()
        num_model.add(Dense(numeric_inputs, input_shape=(numeric_inputs,)))
        total_input_size += numeric_inputs
        feature_models.append(num_model)
        
        merged_model = Sequential()
        if len(feature_models) < 0:
          raise ValueError('No models built, no inputs?')
        elif len(feature_models) == 1:
          merged_model = feature_models[0]
        else:
          merged_model.add(Merge(feature_models, mode='concat', concat_axis=1))
        return merged_model, total_input_size
      
      # We will build in total DEEP_RANGE*WIDE_RANGE models.
      optimizer = hp['optimizer']
      width = hp['width']
      depth = hp['depth']
      activation = hp['activation']
      dropout = 0.2  # hp['dropout']
      batch_size = 128  # hp['batch_size']
      
      model, input_size = init_model(self)
      net_width = input_size * width
      
      # We will add 'depth' layers with 'net_width' neurons.
      for i in range(depth):
        if i == 0 and depth != 1:
          model.add(Dense(net_width, input_shape=(input_size,)))
          model.add(Activation(activation))
          model.add(Dropout(dropout))
        elif i == depth - 1:
          model.add(Dense(len(output_headers), input_shape=(net_width,)))
          model.add(Activation(activation))
        else:
          model.add(Dense(net_width, input_shape=(net_width,)))
          model.add(Activation(activation))
          model.add(Dropout(dropout))

      # No Activation in the end for now... Assuming regression always.
      model.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
      nb_epoch = 20
      if persist:
        nb_epoch = 500
      
      model_name = str(hp).replace('{', '').replace('}', '')
      X_train, Y_train = self.get_data_sets(sample=True)  # Only use a small sample.
      VAL_SPLIT = 0.1  # Split of data to use as validation.
      history = model.fit(X_train, Y_train, 
                          batch_size=batch_size, 
                          nb_epoch=nb_epoch,
                          validation_split=VAL_SPLIT)
      if persist:
        # Save the model for inference purposes.
        from db import persist_keras_models
        persist_keras_models(self.get_handle(), {model_name: model})
      else:
        # Save metrics of this run.
        if model_name not in self.val_losses:
          self.val_losses[model_name] = {}
        for key, val in history.history.iteritems():
          if key in self.val_losses[model_name]:
            self.val_losses[model_name][key].extend(val)
          else:
            self.val_losses[model_name][key] = val
        from db import save_model
        save_model(self)
        
      total_dataset_loss = VAL_SPLIT * history.history['val_loss'][-1] 
      + (1 - VAL_SPLIT) * history.history['loss'][-1]
      return {'loss': total_dataset_loss, 'status': STATUS_OK}
    return run_model_fn
    
  
  # Slices 'data' into lists where each row contains all features. 
  def get_data_sets(self, data=None, string_features=None, sample=False):
    data = data if data else self.data
    string_features = string_features if string_features else self.string_features
    sample_size = len(data.itervalues().next())
    if sample_size > 10000:
      sample_size = sample_size / 10
    X_train = []
    Y_train = []

    for tup in string_features:
      if sample:
        X_train.append(np.array(tup[1][:sample_size]))
      else:
        X_train.append(np.array(tup[1]))

    nums = []
    for idx in xrange(len(data.itervalues().next())):
      row = []
      for header, feature in data.iteritems():
        if header in {tup[0] : None for tup in string_features}:
          continue
        if header.startswith('output_'):
          Y_train.append(feature[idx])
          continue
        row.append(feature[idx])
 
      nums.append(row)
      if sample and len(nums) == sample_size:
        break
      
    X_train.append(np.array(nums))
    
    return X_train, Y_train
      
  # TODO: implement distributed training :O.
  # For now kick-off new process which round-robins doing one epoch each time.
  def start_training(self):
    print 'Starting process'
    train(self.get_handle())
    
  def normalize_float(self, val, header, reverse=False):
    if reverse:
      return val * (self.norms[header][1] - self.norms[header][0]) + self.norms[header][0]
    return (val-self.norms[header][0])/(self.norms[header][1] - self.norms[header][0])

  def normalize_values(self, values):
    for header, column in values.iteritems():
      if self.types[header] != 'str':
        values[header] = [self.normalize_float(float(x), header) for x in column]
  
  def intergerize_string(self, data):
    # Process string features.
    string_features = []
    for header, column in data.iteritems():
      if self.types[header] != 'str':
        continue
      # Every string feature is treated as a list of words.
      word_list = [x.split() for x in data[header]]
      lengths = [len(words) for words in word_list]
      input_size = None
      for tup in self.string_features:
        if header == tup[0]:
          input_size = len(tup[1][0])
          break
      dict_ = self.embedding_dicts[header]
      for idx, words in enumerate(word_list):
        # Strings to integers. Pad sequences with zeros so that all of them have the same size.
        word_list[idx] = pad_sequences([[dict_.get(word, 0) for word in words]], 
                                       maxlen=input_size, padding='post', 
                                       truncating='post')[0].tolist()
      string_features.append((header, word_list))
    
    return string_features
        
  # Values needs to be in the same format as self.data. That is a dictionary of header to value column.
  def infer(self, values):
    from db import load_keras_models
    models = load_keras_models(self.get_handle())
    
    output_headers = [outputs for outputs in self.data.iterkeys() if outputs.startswith('output_')]
    
    self.normalize_values(values)
    string_data = self.intergerize_string(values)
    X_infer, _ = self.get_data_sets(data=values, string_features=string_data)
    print 'X_infer: ' + str(X_infer)
    print 'norms: ' + str(self.norms)
    outputs = {}
    for model_name, model in models.iteritems():
      out = model.predict(X_infer).tolist()
      print 'Raw out: ' + str(out)
      for idx, value in enumerate(out[0]):
        outputs[model_name] = [self.normalize_float(value, output_headers[0], reverse=True)]
    print 'Outputs ' + str(outputs)
    return outputs
    

  def from_json(self, json_str):
    json_obj = json.loads(json_str)
    members = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self,a))]
    for member in members:
      setattr(self, member, json_obj[member])
    
  def to_json(self):
    members = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self,a))]
    return json.dumps({member: getattr(self, member) for member in members})
