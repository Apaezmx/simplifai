import json
import os

def make_register():
    registry = {}
    def registrar(func):
        registry[func.__name__] = func
        return func
    registrar.all = registry
    return registrar
endpoint = make_register()

@endpoint
def name(args):
    return json.dumps({'Name' : 'Powerfull Andres', 'Args' : args})

@endpoint
def upload_csv(args, files):
    if 'upload' not in files:
        print files
        return 'No file specified'
    upload = files['upload']
    name, ext = os.path.splitext(upload.filename)
    if ext != '.csv':
        return 'File extension not recognized'
    save_path = "/tmp/{name}".format(name=name)
    upload.save(save_path)
    model_handle = train(save_path)
    
    return json.dumps({'status': 'OK', 'handle': model_handle})

# Calls endpoint with a map of arguments
def resolve_endpoint(endpoint_str, args, files):
    if endpoint_str in endpoint.all:
        return endpoint.all[endpoint_str](args, files)
    else:
        return 'No such endpoint %s' % endpoint_str
