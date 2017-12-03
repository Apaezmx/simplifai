import os

ENABLE_MEMCACHE = False

class Config():
  """ Static global class holding important parameters.
  """
  mc = {}  # Memchached object shared across threads.
  DISTRIBUTED_HYPEROPT = False  # Needs a MongoDB instance URL to be set.
  ROOT_PATH = os.path.dirname(os.path.realpath(__file__)) 
# Path to the project's directory.
  STATIC_PATH = os.path.dirname(os.path.realpath(__file__)) + '/static'  # Path to static resources.
  tf_server = {}  # Tensorflow local Server.
  
  def set_mc(self, mc):
    self.mc = mc

  def get_mc(self):
    if self.mc and ENABLE_MEMCACHE:
      return self.mc
    return DummyMc()

class DummyMc():
  def set(self, a, b):
    return None
  def get(self, a):
    return None
  def delete(self, a):
    return None

global config
config = Config()  # Singleton class object to be used across the project.
