import sys
from utils.util import inner_sweep_window

i = int(sys.argv[1])
p0 = int(sys.argv[2])
p1 = int(sys.argv[3])
model = sys.argv[4]
config = sys.argv[5]
util = sys.argv[6]
wpath = sys.argv[7]
path = sys.argv[8]
fpath = sys.argv[9]
n_iters = int(sys.argv[10])

p = [p0,p1]

#print(i)
#print(p)
#print(n_iters)

inner_sweep_window(i,p,model,config,util,wpath,path,fpath,n_iters)
