from utils.util import *
import sys

wpath = sys.argv[1]
config = sys.argv[2]
print(wpath)
params,model,config,util,wpath,path,fpath,n_iters = outer_sweep_window( wpath = wpath, config = config )

for i,p in enumerate(params):
    inner_sweep_window(i,p,model,config,util,wpath,path,fpath,n_iters)
