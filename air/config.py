import os

class Config():
  """ Static global class holding important parameters.
  """
  mc = {}  # Memchached object shared across threads.
  ROOT_PATH = os.path.dirname(os.path.realpath(__file__))  # Path to the project's directory.
  STATIC_PATH = os.path.dirname(os.path.realpath(__file__)) + '/static'  # Path to static resources.
  
  def set_mc(self, mc):
    self.mc = mc

  def get_mc(self):
    return self.mc

global config
config = Config()  # Singleton class object to be used across the project.
