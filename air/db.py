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
  model = Model(model_path)
  with open(model_path, 'w+') as f:
      f.write(model.to_json())
  
  return model

def save_model(model):
  with open(model.model_path, 'w+') as f:
      f.write(model.to_json())

def random_hex():
  ran = random.randrange(10**80)
  return "%064x" % ran

