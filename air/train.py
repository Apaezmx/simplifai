import os
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from keras.callbacks import ModelCheckpoint

def info(title):
    print title
    print 'module name: ' +  __name__
    print 'parent process: ' + str(os.getppid())
    print 'process id: ' + str(os.getpid())
    
def layer_choice(num):
  layer_neurons = []
  for i in range(num):
    layer_neurons.append(hp.choice('c'+str(num)+'l'+str(i),range(1, 10)))
  return layer_neurons

def train(handle, train_epochs=50):
  from db import load_keras_models, get_model, save_model
  from model import ModelStatus
  info('Running training on new process')
  air_model = get_model(handle)
  air_model.status = ModelStatus.TRAINING
  save_model(air_model)
  
  fspace = {
    'optimizer': hp.choice('optimzer', ['rmsprop', 'adagrad']),
    'activation': hp.choice('activation', ['relu', 'sigmoid', 'tanh']),
    'layers': hp.choice('layers', [(str(x), layer_choice(x)) for x in range(10)])
  }

  trials = Trials()
  best = fmin(fn=air_model.run_model(), space=fspace, algo=tpe.suggest, max_evals=train_epochs, trials=trials)

  print 'best:', space_eval(fspace, best)

  print 'trials:'
  for trial in trials.trials[:2]:
      print trial
  
  print 'Training finished'
  air_model.status = ModelStatus.TRAINED
  air_model.best_model = best
  save_model(air_model)
  model_fn = air_model.run_model(persist=True)
  model_fn(space_eval(fspace, best))  # Train and persist best model.
