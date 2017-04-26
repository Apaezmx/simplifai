import csv
from datetime import datetime
import json
import random
import os
from model import Model

MODEL_PATH = '/models'

def new_model(current_dir):
  filename = random_hex()
  while os.path.isfile(filename):
      filename = random_hex()

  # Create file
  model_path = current_dir + MODEL_PATH + '/' + filename
  model = Model(path=model_path)
  with open(model_path, 'w+') as f:
      f.write(model.to_json())
  
  return model

def save_model(model):
  with open(model.model_path, 'w+') as f:
      f.write(model.to_json())

def get_model(current_dir, handle):
  model_path = current_dir + MODEL_PATH + "/" + handle
  with open(model_path, "r") as f:
      model = Model()
      model.from_json(f.read())
  return model
  
def parse_val(value):
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
  return value, 'str'
  
def load_csvs(file_list):  
  print file_list
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
              output_headers += 1 if h.startsWith('output_') else 0
            if not output_headers:
              return 'No outputs defined in CSV. Please define columns as outputs by preppending \'output\'.'
          else:
            for idx, value in enumerate(row):
              val, typ = parse_val(value)
              data[headers[idx]].append(val)
              if not types[headers[idx]]:
                types[headers[idx]] = typ
    else:
      print 'WARN: CSV %s not found' % f
    return data, types

def random_hex():
  ran = random.randrange(10**80)
  return "%064x" % ran

