import json
import math
import os
from db import get_model, new_model, save_model, delete_model
from keras.backend.tensorflow_backend import clear_session
from model import ModelStatus

LOSS_LENGTH = 0

def make_register():
  registry = {}
  def registrar(func):
      registry[func.__name__] = func
      return func
  registrar.all = registry
  return registrar
endpoint = make_register()

def handleNaN(val):
  if math.isnan(val):
    return 0
  return val
  
@endpoint
def infer_types(args, files):
  try:
    model = get_model(args['handle'])
  except Exception as e:
    return json.dumps({'status': 'ERROR', 'why': 'Model probably not found '  + str(e)})
  return json.dumps({'status': 'OK', 'types': model.types})

@endpoint
def infer(args, files):
  clear_session()  # Clears TF graphs
  try:
    model = get_model(args['handle'])
  except Exception as e:
    return json.dumps({'status': 'ERROR', 'why': 'Model probably not found ' + str(e)})
    
  if 'values' not in args:
    return json.dumps({'status': 'ERROR', 'why': 'No values specified'})
  
  outputs = model.infer(json.loads(args['values']))
  return json.dumps({'status': 'OK', 'result': outputs})

@endpoint
def train(args, files):
  clear_session()  # Clears TF graphs
  try:
    model = get_model(args['handle'])
  except:
    return json.dumps({'status': 'ERROR', 'why': 'Model probably not found'})
  model.start_training()
  return json.dumps({'status': 'OK', 'handle': model.get_handle()})

@endpoint
def train_status(args, files):
  try:
    model = get_model(args['handle'])
  except:
    return json.dumps({'status': 'ERROR', 'why': 'Model probably not found'})
  if model.status == ModelStatus.TRAINED:
    return json.dumps({'status': 'DONE'})
  losses = {}
  for model_name, value_dict in model.val_losses.iteritems():
    for metric_name, vals in value_dict.iteritems():
      if model_name in losses:
        losses[model_name][metric_name] = [handleNaN(x) for x in vals[-LOSS_LENGTH:]]
      else:
        losses[model_name] = {metric_name: [handleNaN(x) for x in vals[-LOSS_LENGTH:]]}
  return json.dumps({'status': 'OK', 'val_losses': losses})
  
@endpoint
def upload_csv(args, files):
  if 'upload' not in files:
      print 'Files not specified in upload: ' + files
      return 'No file specified'
  upload = files['upload']
  if not upload:
    return 'File not valid'
  name, ext = os.path.splitext(upload.filename)
  if ext != '.csv':
      return 'File extension not recognized'
  save_path = "/tmp/{name}".format(name=upload.filename)
  if not os.path.isfile(save_path):
    upload.save(save_path)
  model = new_model()
  res = model.add_train_file(save_path)
  if res:
    delete_model(model.get_handle())
    return json.dumps({'status': 'ERROR', 'handle': model.get_handle(), 'why': res})
  save_model(model)
  
  return json.dumps({'status': 'OK', 'handle': model.get_handle(), 'types': model.types})

# Calls endpoint with a map of arguments.
def resolve_endpoint(endpoint_str, args, files):
  if endpoint_str in endpoint.all:
      return endpoint.all[endpoint_str](args, files)
  else:
      return 'No such endpoint %s' % endpoint_str
