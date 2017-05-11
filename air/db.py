import csv
import json
import random
import re
import os
from config import config
from datetime import datetime
from keras.models import load_model
from model import Model, ModelStatus

MODEL_PATH = '/models'

def new_model():
  filename = random_hex()
  while os.path.isfile(filename):
      filename = random_hex()

  # Create file
  model_path = config.ROOT_PATH + MODEL_PATH + '/' + filename
  model = Model(path=model_path)
  with open(model_path, 'w+') as f:
      f.write(model.to_json())
  
  return model

def save_model(model):
  with open(model.model_path, 'w+') as f:
      f.write(model.to_json())

def get_model(handle):
  model_path = config.ROOT_PATH + MODEL_PATH + "/" + handle
  with open(model_path, "r") as f:
      model = Model()
      model.from_json(f.read())
  return model
  
def parse_val(value):
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
 
def persist_keras_models(handle, models):
  print str(models)
  model_dir = config.ROOT_PATH + MODEL_PATH
  
  # Clear first all previously persisted models
  for f in os.listdir(model_dir):
    if re.search(handle + "_[0-9]+", f):
        os.remove(os.path.join(model_dir, f))
  for model_name, model in models.iteritems():
    print 'Persisting ' + handle + '_' + unicode(model_name)
    model.save(os.path.join(model_dir, handle + '_' + unicode(model_name)))

def load_keras_models(handle):
  model_dir = config.ROOT_PATH + MODEL_PATH
  models = {}
  
  # Clear first all previously persisted models
  for f in os.listdir(model_dir):
    if re.search(handle + "_.*", f):
      models[f.replace(handle+'_','')] = load_model(os.path.join(model_dir, f))

  return models

def delete_model(handle):
  model_dir = config.ROOT_PATH + MODEL_PATH
  for f in os.listdir(model_dir):
    if re.search(handle + ".*", f):
        os.remove(os.path.join(model_dir, f))
  
def load_csvs(file_list):  
  print 'File of csvs to load ' + unicode(file_list)
  data = {}
  types = {}
  for f in file_list:
    if os.path.isfile(f):
      with open(f, "r") as read_f:
        reader = csv.reader(read_f)
        headers = []
        for row in reader:
          if not headers:
            headers = row
            output_headers = 0
            for h in headers:
              data[h] = []
              types[h] = None
              output_headers += 1 if h.startswith('output_') else 0
            if not output_headers:
              return 'No outputs defined in CSV. Please define columns as outputs by preppending \'output_\'.', ''
          else:
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
    return data, types, norms

def random_hex():
  ran = random.randrange(10**80)
  return "%064x" % ran

