import os
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from keras.callbacks import ModelCheckpoint

def info(title):
    print title
    print 'module name: ' +  __name__
    print 'parent process: ' + str(os.getppid())
    print 'process id: ' + str(os.getpid())

def train(handle, train_epochs=50):
  from db import load_keras_models, get_model, save_model
  from model import ModelStatus
  info('Running training on new process')
  air_model = get_model(handle)
  air_model.status = ModelStatus.TRAINING
  save_model(air_model)
  
  fspace = {
    'optimizer': hp.choice('optimzer', ['rmsprop', 'sgd', 'adagrad']),
    'width': hp.choice('width', range(1,10)),
    'depth': hp.choice('depth', range(1,10)),
    'activation': hp.choice('activation', ['relu', 'sigmoid', 'tanh']),
    'dropout': hp.uniform('dropout', 0.1, 0.4),
    'batch_size': hp.choice('batch_size', [32, 64, 128, 256])
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
