import os
import yaml
import json
import sys

def has_out(path='.'):
    cwd = os.getcwd()
    os.chdir(path)
    check = sorted([f for f in os.listdir('./') if not f.startswith('.')], key=str.lower)
    os.chdir(cwd)
    if 'out' in check:
        return True
    else:
        return False

wpath = os.environ.get('wpath')
#config = os.environ.get('config')

dirs = sys.argv

if len(dirs) < 2:
    sys.exit('please provide at least one path to an experiment folder in the plots directory')


#path = f'{wpath}/{config}'

#with open(path,'r') as file:
#    try:
#        conf = yaml.safe_load(file)
#    except yaml.YAMLError as exc:
#        print(exc)

#plot_path = conf['default']['plot_path']
#experiment_name = conf['default']['experiment_name']

#exp_dir = f'{plot_path}/{experiment_name}'


for d in dirs[1:]:
        
    d = f'{wpath}/{d}'
    meta = {}

    for root, subdirs, files in os.walk(d):

        list_file_path = os.path.join(root, '')
		
        if has_out(path=list_file_path):
            #print('has out')
            with open(f'{list_file_path}/out','r') as file:
                out = json.load(file)
            run = [*out.keys()][0]
            meta[run] = out[run]

        else:
            print(f'{list_file_path} has no out')
		

    with open(f'{d}/out', 'w') as file:
        file.write(json.dumps(meta))
        print(f'> wrote to {d}/out')
