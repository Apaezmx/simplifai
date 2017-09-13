import json
import math
import os
from db import get_model, new_model, save_model, delete_model, clear_thread_cache
from keras.backend.tensorflow_backend import clear_session
from model import ModelStatus

LOSS_LENGTH = 0

def make_register():
  """ Creates a dict registry object to hold all annotated definitions.
  Returns: A dictionary with all definitions in this file.
  """
  registry = {}
  def registrar(func):
      registry[func.__name__] = func
      return func
  registrar.all = registry
  return registrar
  
# Dictionary holding all definitions in this file.
endpoint = make_register()

def handleNaN(val):
  """ Turns all NaN values to 0.
  Returns: 0 if val == NaN, otherwise the value.
  """
  if math.isnan(val):
    return 0
  return val
  
@endpoint
def infer_types(args, files):
  """ Given a model handle, returns a dictionary of column-name to type JSON.
  Args: Model handle
  Returns: a JSON holding the column-name to type map.
  """
  try:
    model = get_model(args['handle'])
  except Exception as e:
    return json.dumps({'status': 'ERROR', 'why': 'Model probably not found '  + str(e)})
  return json.dumps({'status': 'OK', 'types': model.types})

@endpoint
def infer(args, files):
  """ Given a model handle and input values, this def runs the model inference graph and returns the predictions.
  Args: Model handle, input values.
  Returns: A JSON containing all the model predictions.
  """
  clear_session()  # Clears TF graphs.
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
  """ Runs the training for the given model.
  Args: Model handle.
  Returns: A JSON confirming that training has been kicked-off.
  """
  clear_session()  # Clears TF graphs.
  clear_thread_cache()  # We need to clear keras models since graph is deleted.
  try:
    model = get_model(args['handle'])
  except:
    return json.dumps({'status': 'ERROR', 'why': 'Model probably not found'})
  model.start_training()
  return json.dumps({'status': 'OK', 'handle': model.get_handle()})

@endpoint
def train_status(args, files):
  """ Grabs the metrics from disk and returns them for the given model handle.
  Args: Model handle.
  Returns: A JSON with a dictionary of keras_model_name -> metric_name -> list(metric values)
  """
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
  """ Takes in a csv and creates a Model around it.
  CSV needs to have a feature per column. Also needs to have at least one column marked as output by prepending 
  'output_' to the column name (first row in file). Types will be conservatively infered from the input (ie type will be
  string as long as one cell contains a non-numeric character).
  
  Files: Path to tmp CSV file on server (handled by the framework).
  Returns: A JSON with the model handle just created, and the infered feature types.
  """
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

def resolve_endpoint(endpoint_str, args, files):
  """ Reroutes the request to the matching endpoint definition. 
  See make_registrar for more information.
  Params: arguments and files needed to run the endpoint (every endpoint receives both dictionaries). Also receives the
  name of the endpoint.
  Returns: The output of the endpoint.
  """
  if endpoint_str in endpoint.all:
      return endpoint.all[endpoint_str](args, files)
  else:
      return 'No such endpoint %s' % endpoint_str
