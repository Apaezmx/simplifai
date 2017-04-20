import os

class Config():
  ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
  STATIC_PATH = os.path.dirname(os.path.realpath(__file__)) + '/static'
  
global config
config = Config()
