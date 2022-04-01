from utils.util import inner_sweep_window
import sys

i = sys.argv[1]
p = sys.argv[2]
file_idxs = sys.argv[3]
model = sys.argv[4]
columns = sys.argv[5]
n_files = sys.argv[6]
classDict = sys.argv[7]
n_iters = sys.argv[8]
sweep = sys.argv[9]
path = sys.argv[10]

print(i)
print(p)
print(file_idxs)
print(model)
print(columns)
print(n_files)
print(classDict)
print(n_iters)
print(sweep)
print(path)

#inner_sweep_window( i,p,S,file_idxs,model,columns,n_files,classDict,n_iters,sweep,path )
