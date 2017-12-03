import os
from config import config
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from hyperopt.mongoexp import MongoTrials
from keras.callbacks import ModelCheckpoint

def info(title):
    """ Prints out some threads stats. """
    print title
    print 'module name: ' +  __name__
    print 'parent process: ' + str(os.getppid())
    print 'process id: ' + str(os.getpid())

def gen_layer(num_ls, num_l, width):
  """ Given the number of layers, the layer number and width, returns a choice of activation for the layer. """
  return (width, hp.choice('layer activation ' + str(num_ls) + str(num_l) + str(width), ['relu', 'sigmoid', 'tanh', 'linear']))#NEQP (Yo creo que esto podemos optimizarlo todavia mas dependiendo del tipo de data que se quiera entrenar, ya que hay algunas activaciones que no tienen mucha utilidad para cierto tipo de data)

def layer_choice(num):
  """ Given the number of layers, returns a choice of differing width and activations layers. """
  layer_neurons = []
  for i in range(num):
    # Choose number of neurons from 1 - 10.
    layer_neurons.append(hp.choice('num_layers ' + str(num) + ' layer ' + str(i), [('num_layers ' + str(num) + ' layer ' + str(i) + ' width ' + str(x), gen_layer(num, i, x)) for x in range(1,11)]))#NEQP (Tambien siento que de 1 a 10 neuronas igual y no es la unica opcion que deberiamos de tener, ya que para cierto tipo de data muy compleja, mas que un numero elevado de layers tal vez necesitemos no tantas layers pero mas neuronas en cada layer)
  return layer_neurons

def train(air_model, train_epochs=20):
  """ Runs TPE black box optimization of the neural network to use.
  After evaluating all points, it saves the best model to disk and sets the status flag as TRAINED.
  """
  from db import get_model, save_model
  from model import ModelStatus
  info('Running training on new process')
  air_model.status = ModelStatus.TRAINING
  save_model(air_model)
  
  fspace = {
    'optimizer': hp.choice('optimzer', ['rmsprop', 'adagrad']), #NEQP (Supongo que si, pero es a proposito que diga 'optimzer'?)
    'layers': hp.choice('layers', [(str(x), layer_choice(x)) for x in range(10)])  # Choose from 0 to 9 layers.
  }
  
  if config.DISTRIBUTED_HYPEROPT:
    # TODO: Probably not send all model from json. Just send the ids and make the worker fetch it from the DB.
    fspace['model_json'] = air_model.to_json()
    trials = MongoTrials('mongo://localhost:27017/testdb/jobs', exp_key='userid.trainingid', workdir='/home/paezand/pusher/bottle_air')
    best = fmin(fn=run_model_fn, space=fspace, trials=trials, algo=tpe.suggest, max_evals=train_epochs)
    # Run workers with
    # hyperopt-mongo-worker --mongo=$mongodbURL/testdb --poll-interval=0.1 --workdir=$bottle_air_dir
  else:
    trials = Trials() #NEQP (Checaste la opcion de hacer parallel search con MongoDB?)
    best = fmin(fn=air_model.run_model(), space=fspace, algo=tpe.suggest, max_evals=train_epochs, trials=trials)

  print 'best:', space_eval(fspace, best)

  print 'trials:'
  for trial in trials.trials[:2]:
    print trial
  
  model_fn = air_model.run_model(persist=True)
  model_fn(space_eval(fspace, best))  # Train and persist best model.

  print 'Training finished'
  air_model.status = ModelStatus.TRAINED
  air_model.best_model = best
  save_model(air_model)


## TRAINING FUNCTION FOR DISTRIBUTED HYPEROPT ##
def run_model_fn(hp):
  """ Definition to be evaluated by the black box optimizer.
    Params: hyperparameter dictionary.
  """
  from model import Model
  import json
  import numpy as np
  from keras.models import Sequential
  from keras.layers import Embedding, Dense, Activation, Merge, Flatten, Dropout
  from keras.preprocessing.sequence import pad_sequences
  from keras_utils import single_activation
  import tensorflow as tf
  import os
  cwd = os.getcwd()
  print cwd
  air_model = Model()
  air_model.from_json(hp["model_json"])
  output_headers = [outputs for outputs in air_model.data.iterkeys() if outputs.startswith('output_')]
  if not output_headers:
    raise ValueError('No outputs defined!')
  
  # Process string features.
  if not air_model.string_features: 
    air_model.string_features = []
    for header, typ in air_model.types.iteritems():
      if typ != 'str':
        continue
      # Every string feature is treated as a list of words.
      word_list = [x.split() for x in air_model.data[header]]
      dict_, _ = air_model.process_text_feature(word_list)
      assert len(dict_) > 0, 'Dict is empty.'
      air_model.embedding_dicts[header] = dict_ #NEQP (Si haces un nuevo dict para cada columna de strings, no hay entonces idx que se repiten para diferentes palabras?)
      lengths = [len(words) for words in word_list]
      lengths.sort()
      input_size = lengths[int(np.round((len(lengths)-1) * 0.95))] #NEQP (Para que es este calculo?)
      if input_size == 0:
        print 'WARNING: input_size is 0 for ' + header
      input_size = 1
      for idx, words in enumerate(word_list):
        # Strings to integers. Pad sequences with zeros so that all of them have the same size.
        word_list[idx] = pad_sequences([[dict_[word] for word in words]], 
                  maxlen=input_size, padding='post', 
                   truncating='post')[0].tolist() #NEQP (Y esto que pex?)
      air_model.string_features.append((header, word_list))
  
  # Build models.
  # Merge all inputs into one model.
  def init_model(air_model):
    feature_models = []
    total_input_size = 0
    i = 0
    for tup in air_model.string_features:
      header = tup[0]
      word_list = tup[1]
      sequence_length = len(word_list[0])
      embedding_size = int(np.round(np.log10(len(air_model.embedding_dicts[header]))))
      embedding_size = embedding_size if embedding_size > 0 else 1
      model = Sequential(name='str_model_' + str(len(feature_models)))
      model.add(Embedding(len(air_model.embedding_dicts[header].keys()), embedding_size, input_length=sequence_length, name='embedding_model_' + str(len(feature_models))))
      model.add(Flatten(name='flatten_model_' + str(len(feature_models))))
      total_input_size += embedding_size * len(word_list[0]) #NEQP (Si hay un embedding por palabra: realmente los embeddings podran generar un vector de significado? Y, no se le da mucho peso a las strings por sobre los integers?)
      feature_models.append(model)
    
    numeric_inputs = len(air_model.data) - len(air_model.string_features) - len(output_headers)
    if numeric_inputs:
      num_model = Sequential(name='num_model_' + str(len(feature_models)))
      num_model.add(Dense(numeric_inputs, input_shape=(numeric_inputs,), name='dense_model_' + str(len(feature_models))))
      total_input_size += numeric_inputs
      feature_models.append(num_model)
    
    merged_model = Sequential()
    if len(feature_models) < 1:
      raise ValueError('No models built, no inputs?')
    elif len(feature_models) == 1:
      merged_model = feature_models[0]
    else:
      merged_model.add(Merge(feature_models, mode='concat', concat_axis=1))
    return merged_model, total_input_size
  
  # We will build in total DEEP_RANGE*WIDE_RANGE models.
  optimizer = hp['optimizer']
  layers = hp['layers']
  dropout = 0.2  # hp['dropout'] #NEQP (No estaria bueno igual que hyperopt tambien optimizara estos hyperparamenters?)
  batch_size = 128  # hp['batch_size']
  
  model, input_size = init_model(air_model)
  
  # We will add 'depth' layers with 'net_width' neurons.
  depth = len(layers[1])
  for i in range(depth):
    layer_activation = layers[1][i][1][1]
    layer_width = layers[1][i][1][0] #NEQP (Creo que es una buena practica variar el width de las layers. Normalmente se usan variaciones tipo 10-20-40-20-10)
    if i == 0 and depth != 1:
      model.add(Dense(layer_width, input_shape=(input_size,), name='layer_model_' + str(i)))
      model.add(Activation(layer_activation))
      model.add(Dropout(dropout))
    elif i == depth - 1:
      model.add(Dense(len(output_headers), input_shape=(len(layers[1][i-1][1]),), name='layer_model_' + str(i)))
    else:
      model.add(Dense(layer_width, input_shape=(len(layers[1][i-1][1]),), name='layer_model_' + str(i)))
      model.add(Activation(layer_activation))
      model.add(Dropout(dropout))
  
  if not depth:
    model.add(Dense(len(output_headers), input_shape=(input_size,), name='layer_model_0'))
  # No Activation in the end for now... Assuming regression always.
  model.compile(loss='mse',
    optimizer=optimizer,
    metrics=['accuracy'])
  nb_epoch = 30
  
  model_name = str(hp).replace('{', '').replace('}', '')
  X_train, Y_train = air_model.get_data_sets(sample=True)  # Only use a small sample.

  VAL_SPLIT = 0.1  # Split of data to use as validation.
  print 'Sizes: ' + str(len(X_train)) + ', ' + str(X_train[0].shape) + ' ' + str(len(Y_train))
  with tf.Session() as sess:
    history = model.fit(X_train, Y_train, 
            batch_size=batch_size, #NEQP (Entonces X_train como esta organizado? Cual es su forma?)
            nb_epoch=nb_epoch,
            shuffle=True,
            validation_split=VAL_SPLIT)
    
  total_dataset_loss = VAL_SPLIT * history.history['val_loss'][-1] 
  + (1 - VAL_SPLIT) * history.history['loss'][-1]
  return {'loss': total_dataset_loss, 'status': STATUS_OK}

