import json
import os
from db import get_model, new_model, save_model
from config import config

def make_register():
  registry = {}
  def registrar(func):
      registry[func.__name__] = func
      return func
  registrar.all = registry
  return registrar
endpoint = make_register()

@endpoint
def train(args, files):
  try:
    model = get_model(config.ROOT_PATH, args['handle'])
  except:
    return json.dumps({'status': 'ERROR', 'why': 'Model probably not found'})
  model.start_training()
  return json.dumps({'status': 'OK', 'handle': model.get_handle()})

@endpoint
def train_status(args, files):
  return json.dumps({'status': 'OK'})
  
@endpoint
def upload_csv(args, files):
  if 'upload' not in files:
      print files
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
  model = new_model(config.ROOT_PATH)
  model.add_train_file(save_path)
  save_model(model)
  
  return json.dumps({'status': 'OK', 'handle': model.get_handle(), 'types': model.types})

# Calls endpoint with a map of arguments
def resolve_endpoint(endpoint_str, args, files):
  if endpoint_str in endpoint.all:
      return endpoint.all[endpoint_str](args, files)
  else:
      return 'No such endpoint %s' % endpoint_str
