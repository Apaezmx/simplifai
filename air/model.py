class ModelStatus():
  NULL = 0
  CREATED = 1
    
class Model():
  model_path = ''
  train_files = []
  status = ModelStatus.NULL

  def __init__(self, path):
    self.model_path = path
    self.status = ModelStatus.CREATED
    
  def get_handle(self):
    return path.split('/')[-1]
    
  def add_train_file(filename):
    train_files.append(filename)

  def from_json(self, json):
    self.train_files = json['train_files']
    self.status = json['status']
    self.model_path = json['model_path']
    
  def to_json(self):
    members = [a for a in dir(self) if not a.startswith('__') and not callable(getattr(self,a))]
    return json.dumps({member: getattr(self, member) for member in members})
