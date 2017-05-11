import os
from keras.callbacks import ModelCheckpoint

def info(title):
    print title
    print 'module name: ' +  __name__
    print 'parent process: ' + str(os.getppid())
    print 'process id: ' + str(os.getpid())
    
EPOCHS_PER_MODEL = 100
TOTAL_EPOCHS = 1000

def train(handle, train_epochs=30):
  from db import load_keras_models, get_model, save_model
  from model import ModelStatus
  info('Running training on new process')
  air_model = get_model(handle)
  air_model.status = ModelStatus.TRAINING
  save_model(air_model)
  models = load_keras_models(handle)
  X_train, Y_train = air_model.get_data_sets()
  
  epoch = 0
  go_crit = True
  while go_crit:
    print 'Epoch: ' + str(epoch) + ' num_models: ' + str(len(models))
    epoch += 1
    all_ended = True
    for model_name, model in models.iteritems():
      print 'Model: ' + model_name
      checkpoint = ModelCheckpoint(air_model.model_path + '_' + model_name)
      history = model.fit(X_train, Y_train, 
                          batch_size=100, 
                          nb_epoch=EPOCHS_PER_MODEL,
                          callbacks=[checkpoint], 
                          validation_split=0.2)
      for layer in model.layers:
        weights = layer.get_weights()
      if model_name not in air_model.val_losses:
        air_model.val_losses[model_name] = {}
      for key, val in history.history.iteritems():
        if key in air_model.val_losses[model_name]:
          air_model.val_losses[model_name][key].extend(val)
        else:
          air_model.val_losses[model_name][key] = val
      if history.history['loss'][0] > 0.01:
        all_ended = False 
      save_model(air_model)
    go_crit = not all_ended and epoch < TOTAL_EPOCHS / EPOCHS_PER_MODEL
  
  print 'Training finished'
  air_model.stats = ModelStatus.TRAINED
  save_model(air_model)
