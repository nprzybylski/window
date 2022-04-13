from utils.util import *
import sys
import os

#print(sys.argv)

wpath = sys.argv[1]
config = sys.argv[2]
dpath = os.environ.get('dpath')
model = os.environ.get('model')

#print(wpath)
params,model,config,util,wpath,path,fpath,n_iters = outer_sweep_window( wpath = wpath, config = config, model = model )


for i,p in enumerate(params):
    print(f'ITER {i}: {n_iters}')
    cmd = f'sbatch --export=i={i},p0={p[0]},p1={p[1]},model={model},config={config},util={util},wpath={wpath},path={path},fpath={fpath},n_iters={n_iters},dpath={dpath} inner.sh'

#{i} {p[0]} {p[1]} {model} {config} {util} {wpath} {path} {fpath} {n_iters}'

    print(f'CMD: {cmd}')
    os.system(cmd)
