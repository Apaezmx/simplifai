import json
import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import Embedding, Dense, Activation, Merge, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, SGD, Adagrad
from keras.preprocessing.sequence import pad_sequences
from multiprocessing import Process
from train import train

import numpy as np
import os
import os.path

# Layers for now will have a constant width across layers. It has a constant number of width specified by WIDE_RANGE
# plus a dynamic number which depends on the number of inputs. That is we will add WIDE_RATIO neurons for each input.
WIDE_RANGE = 1
DEEP_RANGE = 1
WIDE_RATIO = 5
MIN_DEPTH = 2

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
  
  # Same as data but with integerized strings for string features.
  string_features = []
  
  # FeatureName keyed dict for string feature containing the dictionary word->count.
  embedding_dicts = {}
  
  # Keras models history.
  val_losses = {}

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

  def build_models(self):
    start_depth = len(self.data) + MIN_DEPTH
    start_width = len(self.types)
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
      self.string_features.append({header: word_list})
    
    # Build models.
    # Merge all inputs into one model.
    def init_model(self):
      feature_models = []
      total_input_size = 0
      for dict_ in self.string_features:
        header = dict_.iterkeys().next()
        word_list = dict_.itervalues().next()
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
      num_model.add(Activation('relu'))
      num_model.add(Dropout(0.2))
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
    keras_models = {}
    optimizers = [(RMSprop(), 'RMSprop'), (SGD(clipnorm=1.), 'SGD'), (Adagrad(), 'Adagrad')]
    for optimizer in optimizers:
      for depth in range(start_depth, start_depth + DEEP_RANGE):
        for width in range(start_width, start_width + WIDE_RANGE):
          model, input_size = init_model(self)
          net_width = width + input_size * WIDE_RATIO
          
          # We will add 'depth' layers with 'net_width' neurons.
          for i in range(depth):
            if i == 0:
              model.add(Dense(net_width, input_shape=(input_size,)))
              model.add(Activation('relu'))
              model.add(Dropout(0.2))
            elif i == depth - 1:
              model.add(Dense(len(output_headers), input_shape=(net_width,)))
              model.add(Activation('relu'))
              model.add(Dropout(0.2))
            else:
              model.add(Dense(net_width, input_shape=(net_width,)))
              model.add(Activation('relu'))
              model.add(Dropout(0.2))
          
          # Add a last 'scaler' for output normalization.
          model.add(Dense(len(output_headers), input_shape=(len(output_headers),)))

          # No Activation in the end for now... Assuming regression always.
          model.summary()
          model.compile(loss='mse',
                optimizer=optimizer[0],
                metrics=['accuracy'])
          
          keras_models[str(net_width) + 'x' + str(depth) + '-' + optimizer[1]] = model
    
    from db import persist_keras_models
    persist_keras_models(self.get_handle(), keras_models)
  
  # Slices 'data' into lists where each row contains all features. 
  def get_data_sets(self, data=self.data, string_features=self.string_features):
    print len(data.itervalues().next())
    if len(string_features) > 0:
      print len(string_features[0].itervalues().next())

    X_train = []
    Y_train = []

    for dict_ in string_features:
      X_train.append(np.array(dict_.itervalues().next()))

    nums = []
    for idx in xrange(len(data.itervalues().next())):
      row = []
      for header, feature in data.iteritems():
        if header in {h.iterkeys().next() : h for h in string_features}:
          continue
        if header.startswith('output_'):
          Y_train.append(feature[idx])
          continue
        row.append(feature[idx])
 
      nums.append(row)
      
    X_train.append(np.array(nums))
     
    return X_train, Y_train
      
  # TODO: implement distributed training :O.
  # For now kick-off new process which round-robins doing one epoch each time.
  def start_training(self):
    print 'Starting process'
    train(self.get_handle())
    
  def normalize_values(self, values):
    for header, column in values.iteritems():
      if types[header] != 'str':
        values[header] = [2*(x-norms[header][0])/(norms[header][1] - norms[header][0]) - 1 for x in column]
  
  def intergerize_string(self, data):
    # Process string features.
    string_features = []
    for dict_ in self.string_features:
      if self.types[ != 'str':
        continue
      # Every string feature is treated as a list of words.
      word_list = [x.split() for x in data[header]]
      lengths = [len(words) for words in word_list]
      input_size = len(self.string_features[header][0])
      for idx, words in enumerate(word_list):
        # Strings to integers. Pad sequences with zeros so that all of them have the same size.
        word_list[idx] = pad_sequences([[dict_[word] for word in words]], 
                                       maxlen=input_size, padding='post', 
                                       truncating='post')[0].tolist()
      self.string_features.append({header: word_list})
        
    
  def infer(self, values):
    if self.status != ModelStatus.TRAINING or self.status != ModelStatus.TRAINED:
      print 'Model not ready for inference...'
      return 'Model not ready for inference...'
    from db import load_keras_models, get_model, save_model
    air_model = get_model(self.handle)
    models = load_keras_models(self.handle)
    
    normalize_values(values)
    string_data = intergerize_string(data)
    X_infer, _ = get_data_sets(
    

  def from_json(self, json_str):
    json_obj = json.loads(json_str)
    members = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self,a))]
    for member in members:
      setattr(self, member, json_obj[member])
    
  def to_json(self):
    members = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self,a))]
    return json.dumps({member: getattr(self, member) for member in members})
