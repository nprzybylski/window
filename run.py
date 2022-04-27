from utils.util import *
import sys
import os

#print(sys.argv)

wpath = sys.argv[1]
config = sys.argv[2]

dpath = os.environ.get('dpath')
model = os.environ.get('model')
vpath = os.environ.get('vpath')

#print(wpath)
if len(sys.argv) > 3:
    run = sys.argv[3]
    params,model,config,util,wpath,path,fpath,n_iters = outer_sweep_window( wpath = wpath, config = config, model = model, run = run, vpath = vpath )
else:
    params,model,config,util,wpath,path,fpath,n_iters = outer_sweep_window( wpath = wpath, config = config, model = model, vpath = vpath )
#print(wpath, config, dpath, model, vpath)

for i,p in enumerate(params):
#    print(f'ITER {i}: {n_iters}')
    cmd = f'sbatch --export=i={i},p0={p[0]},p1={p[1]},model={model},config={config},util={util},wpath={wpath},path={path},fpath={fpath},n_iters={n_iters},dpath={dpath},vpath={vpath} inner.sh'

#{i} {p[0]} {p[1]} {model} {config} {util} {wpath} {path} {fpath} {n_iters}'

    print(f'CMD: {cmd}')
    os.system(cmd)
