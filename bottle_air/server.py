"""
Imports for bottle
"""
import argparse
import bottle
from bottle import request, run, static_file
"""
Imports for API
"""
import tensorflow as tf
from db import get_model, new_model, save_model, delete_model, clear_thread_cache
from keras.backend.tensorflow_backend import clear_session
from model import ModelStatus
"""
Other imports
"""
from config import config
import os
import json
import math

app = bottle.Bottle()
plugin = bottle.ext.memcache.MemcachePlugin(servers=['localhost:11211'])
app.install(plugin)

LOSS_LENGTH = 0

def handleNaN(val):
  """ Turns all NaN values to 0.
  Returns: 0 if val == NaN, otherwise the value.
  """
  if math.isnan(val):
    return 0
  return val

global mem_store

@app.error(404)
def error404(error):
  print error
  return 'Nothing here, sorry'
  
"""
Static Files
"""
@app.route('/infer')
def serve_infer():
  return static_file('infer.html', root=config.STATIC_PATH)
  
@app.route('/')
def serve_index():
  return static_file('index.html', root=config.STATIC_PATH)

@app.route('/<filepath:path>')
def server_static(filepath):
  return static_file(filepath, root=config.STATIC_PATH)
  
"""
API Methods
"""
@app.post('/csv/upload')
def upload_csv(mc):
  """ Takes in a csv and creates a Model around it.
  CSV needs to have a feature per column. Also needs to have at least one column marked as output by prepending 
  'output_' to the column name (first row in file). Types will be conservatively infered from the input (ie type will be
  string as long as one cell contains a non-numeric character).
  
  Files: Path to tmp CSV file on server (handled by the framework).
  Returns: A JSON with the model handle just created, and the infered feature types.
  """
  files = {k: v for k, v in request.files.iteritems()}
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
  
@app.post('/model/train')
def train():
  """ Runs the training for the given model.
  Args: Model handle.
  Returns: A JSON confirming that training has been kicked-off.
  """
  args =  {k: v for k, v in request.forms.iteritems()}
  clear_session()  # Clears TF graphs.
  clear_thread_cache()  # We need to clear keras models since graph is deleted.
  try:
    model = get_model(args['handle'])
  except:
    return json.dumps({'status': 'ERROR', 'why': 'Model probably not found'})
  model.start_training()
  return json.dumps({'status': 'OK', 'handle': model.get_handle()})
  
@app.post('/model/status')
def train_status():
  """ Grabs the metrics from disk and returns them for the given model handle.
  Args: Model handle.
  Returns: A JSON with a dictionary of keras_model_name -> metric_name -> list(metric values)
  """
  args = {k: v for k, v in request.forms.iteritems()}
  try:
    model = get_model(args['handle'])
  except:
    return json.dumps({'status': 'ERROR', 'why': 'Model probably not found'})
  if not model:
    return json.dumps({'status': 'ERROR', 'why': 'Concurrency error'})
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
  
@app.post('/types/infer')
def infer_types():
  """ Given a model handle, returns a dictionary of column-name to type JSON.
  Args: Model handle
  Returns: a JSON holding the column-name to type map.
  """
  args = {k: v for k, v in request.forms.iteritems()}
  try:
    model = get_model(args['handle'])
  except Exception as e:
    return json.dumps({'status': 'ERROR', 'why': 'Model probably not found '  + str(e)})
  return json.dumps({'status': 'OK', 'types': model.types})

@app.post('/inference/make')
def infer():
  """ Given a model handle and input values, this def runs the model inference graph and returns the predictions.
  Args: Model handle, input values.
  Returns: A JSON containing all the model predictions.
  """
  args = {k: v for k, v in request.forms.iteritems()}
  print(args)
  # clear_session()  # Clears TF graphs.
  clear_session()  # Clears TF graphs.
  clear_thread_cache()  # We need to clear keras models since graph is deleted.
  try:
    model = get_model(args['handle'])
  except Exception as e:
      return json.dumps({'status': 'ERROR', 'why': 'Infer: Model probably not found ' + str(e)})
    
  if 'values' not in args:
    return json.dumps({'status': 'ERROR', 'why': 'No values specified'})
  print(args['handle'])
  print(args['values'])
  outputs = model.infer(json.loads(args['values']))
  return json.dumps({'status': 'OK', 'result': outputs})


"""
Run server
"""
parser = argparse.ArgumentParser(description='Running air server')
parser.add_argument('--root_path', type=str, help='filepath to server root')
args = parser.parse_args()

config.tf_server = tf.train.Server.create_local_server()
run(app, reloader=True, host='localhost', port=8012, server='cherrypy')
