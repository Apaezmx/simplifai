import os

class Config():
  mc = {}
  ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
  STATIC_PATH = os.path.dirname(os.path.realpath(__file__)) + '/static'
  
  def set_mc(self, mc):
    self.mc = mc

  def get_mc(self):
    return self.mc

global config
config = Config()
