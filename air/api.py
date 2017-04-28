import json
import os
from db import get_model, new_model, save_model, delete_model

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
    model = get_model(args['handle'])
  except:
    return json.dumps({'status': 'ERROR', 'why': 'Model probably not found'})
  if not model.string_features:
    model.build_models()
    save_model(model)
  model.start_training()
  return json.dumps({'status': 'OK', 'handle': model.get_handle()})

@endpoint
def train_status(args, files):
  try:
    model = get_model(args['handle'])
  except:
    return json.dumps({'status': 'ERROR', 'why': 'Model probably not found'})
  return json.dumps({'status': 'OK', 'val_losses': model.val_losses})
  
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
