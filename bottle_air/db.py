import bottle
import bottle.ext.memcache
import csv
import json
import random
import re
import os
import zlib
from config import config
from datetime import datetime
from keras.models import load_model, model_from_json
from model import Model, ModelStatus

MODEL_PATH = '/models'
HANDLE_LENGTH = 10
keras_cache = {}  # Local thread memory cache.
COMPRESSION_LEVEL = 9

def clear_thread_cache():
  keras_cache = {}

def handle2path(handle):
  """ Translates a handle to a full path on the FS. """
  return config.ROOT_PATH + MODEL_PATH + "/" + handle

def path2handle(path):
  """ Translates a full path to a handle. """
  return path.split('/')[-1]
  
def list_models(ONLY_TRAINED=1):
  """ Lists all current models."""
  models = []
  
  model_dir = config.ROOT_PATH + MODEL_PATH
  
  for name in os.listdir(model_dir):
    if "_" not in name:
      with open(model_dir + '/' + name, 'r') as f:
        model = Model()
        model.from_json(f.read())
        if ONLY_TRAINED and model.status != ModelStatus.TRAINED:
          continue
        models.append(model)

  return models

def new_model():
  """ Construcs an empty Model assiging it a new random hex ID and persiting it to disk. 
  Returns: The Model instance.
  """
  filename = random_hex()
  while os.path.isfile(filename):
    filename = random_hex()

  # Create file
  model_path = config.ROOT_PATH + MODEL_PATH + '/' + filename
  model = Model(path=model_path)
  with open(model_path, 'w+') as f:
    json = model.to_json()
    f.write(json)
    f.close()
    config.get_mc().set(path2handle(model_path), zlib.compress(json, COMPRESSION_LEVEL))
  
  return model

def save_model(model):
  """ Saves the given model to disk. """
  with open(model.model_path, 'w+') as f:
    json = model.to_json()
    f.write(json)
    f.close()
    config.get_mc().set(path2handle(model.model_path), zlib.compress(json, COMPRESSION_LEVEL))

def get_model(handle):
  """ Fetches the model from memory or disk with a matching handle.
  Returns: The Model instance if the model is found, None otherwise.
  """
  mem_try = config.get_mc().get(handle)
  if mem_try:
    m = Model()
    print "From mc"
    m.from_json(zlib.decompress(mem_try))
    if m.data:
      return m
  model_path = config.ROOT_PATH + MODEL_PATH + "/" + handle
  retries = 5
  while retries:
    try:
      with open(model_path, "r") as f:
        model = Model()
        model.from_json(f.read())
        config.get_mc().set(handle, zlib.compress(model.to_json(), COMPRESSION_LEVEL))
        return model
    except Exception as e:
      print "ERROR: Could not load " + handle + " model." + str(e)
    retries -= 1
  return None
  
def parse_val(value):
  """ Infers the type of the value by trying to parse it to different formats.
  Returns: The parse value and the type.
  """
  if not value:
    return value, None
  tests = [
      # (Type, Test)
      ('int', int),
      ('float', lambda value: float(value.replace('$','').replace(',',''))),
      ('date', lambda value: datetime.strptime(value, "%Y/%m/%d")),
      ('time', lambda value: datetime.strptime(value, "%H:%M:%S"))
  ]

  for typ, test in tests:
    try:
      v = test(value)
      # Treat date and time as strings, just replace separator with spaces.
      if typ == 'date':
        v = value.split('/')
        typ = 'str'
      elif typ == 'time':
        v = value.split(':')
        typ = 'str'
      return v, typ
    except ValueError:
      continue
  # No match
  return value.decode('utf-8', 'ignore'), 'str'
 
def persist_keras_model(handle, model):
  """ Persists a keras model to disk.
  """
  model_dir = config.ROOT_PATH + MODEL_PATH
  
  # Clear first all previously persisted models.
  for f in os.listdir(model_dir):
    if re.search(handle + "_keras", f):
        os.remove(os.path.join(model_dir, f))
  name = handle + '_keras'
  print 'Persisting ' + name
  model.save(os.path.join(model_dir, name))

def _load_keras_model(handle):
  """ Loads a keras model from disk.
  Returns: The keras model instance if found, None otherwise.
  """
  name = handle + '_keras'
  print 'load ' + name
  model_dir = config.ROOT_PATH + MODEL_PATH
  
  for f in os.listdir(model_dir):
    if re.search(name, f):
      model = load_model(os.path.join(model_dir, f))
      model.model._make_predict_function()
      print 'From disk'
      return model

def load_keras_model(handle):
  """ Loads a keras model from cache or disk.
  Returns: The keras model instance.
  """
  if False and handle in keras_cache:
    print 'From thread cache'
    return keras_cache[handle]
  model = _load_keras_model(handle)
  keras_cache[handle] = model
  return model

def delete_model(handle):
  """ Deletes all models with the given handle if found. 
  Returns: The number of models deleted.
  """
  model_dir = config.ROOT_PATH + MODEL_PATH
  num_models = 0
  for f in os.listdir(model_dir):
    if re.search(handle + ".*", f):
        os.remove(os.path.join(model_dir, f))
        config.get_mc().delete(handle)
        num_models += 1
        
  return num_models
        
  
def load_csvs(file_list):  
  """ Loads csv from files and returns the parsed value dictionary.
  Params: The list of files.
  Returns: Three dictionaries. The first is feature-name -> value_list, the second one feature_name -> type and the
  third one feature_name -> [min_value, max_value] if applies.
  """
  print 'File of csvs to load ' + unicode(file_list)
  data = {}
  types = {}
  for f in file_list:
    if os.path.isfile(f):
      with open(f, "r") as read_f:
        reader = csv.reader(read_f)
        headers = []
        for row in reader:
          if not headers:  # If first row, load the headers assuming they are contained in the first row.
            headers = row
            output_headers = 0
            for h in headers:
              data[h] = []
              types[h] = None
              output_headers += 1 if h.startswith('output_') else 0
            if not output_headers:
              return 'No outputs defined in CSV. Please define columns as outputs by preppending \'output_\'.', ''
          else:  # If not first row, parse values assuming the headers dictionary has been already filled.
            assert len(row) == len(headers), 'Uneven rows and headers at row ' + str(row) + ' with headers ' + str(headers)
            for idx, value in enumerate(row):
              val, typ = parse_val(value)
              data[headers[idx]].append(val)
              # If we find an entry which is string, then move all the column to be string.
              if not types[headers[idx]] or typ == 'str':
                types[headers[idx]] = typ
    else:
      print 'WARN: CSV %s not found' % f
    
    # Fix '' values, and standardize formats.
    for header, column in data.iteritems():
      for idx, value in enumerate(column):
        if not value:
          data[header][idx] = 0 if types[header] != 'str' else ''
        else:
          data[header][idx] = unicode(data[header][idx]) if types[header] == 'str' else data[header][idx]
    
    # Normalize numeric inputs to -1 to 1.
    norms = {}
    for header, column in data.iteritems():
      if types[header] != 'str':
        floor = float(min(column))
        ceil = float(max(column))
        norms[header] = (floor, ceil)
        data[header] = [(x-floor)/(ceil - floor) for x in column]
    
    # Run some last verifications so that all features have the same amount of rows.
    length = 0
    for header, column in data.iteritems():
      if not length:
        length = len(column)
        continue
      if length != len(column):
        raise ValueError(header + ' column has different lengths: ' + str(length) + ' ' + str(len(column)))
    return data, types, norms

def random_hex():
  """ Creates a random hex string ID """
  ran = random.randrange(16**HANDLE_LENGTH)
  return "%010x" % ran

