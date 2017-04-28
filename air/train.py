import os
from keras.callbacks import ModelCheckpoint

def info(title):
    print title
    print 'module name: ' +  __name__
    print 'parent process: ' + str(os.getppid())
    print 'process id: ' + str(os.getpid())

def train(handle, train_epochs=30):
  from db import load_keras_models, get_model, save_model
  from model import ModelStatus
  info('Running training on new process')
  air_model = get_model(handle)
  air_model.status = ModelStatus.TRAINING
  models = load_keras_models(handle)
  X_train, Y_train = air_model.get_data_sets()
  
  epoch = 0
  go_crit = True
  while go_crit:
    print 'Epoch: ' + str(epoch)
    epoch += 1
    all_ended = True
    for idx, model in enumerate(models):
      checkpoint = ModelCheckpoint(air_model.model_path + '_' + str(idx))
      history = model.fit(X_train, Y_train, batch_size=32, nb_epoch=5, callbacks=[checkpoint], validation_split=0.1)
      while len(air_model.val_losses) <= idx:
        air_model.val_losses.append({})
      for key, val in history.history.iteritems():
        if key in air_model.val_losses[idx]:
          air_model.val_losses[idx][key].extend(val)
        else:
          air_model.val_losses[idx][key] = val
      if history.history['loss'][0] > 1:
        all_ended = False 
      save_model(air_model)
    go_crit = not all_ended
