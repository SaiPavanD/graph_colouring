import numpy as np
from scipy import sparse
import sys

if(len(sys.argv)!=3):
    print 'Usage : python gen_csr <ip_file> <op_file>'
    sys.exit(0)

num_nodes = 0
num_edges = 0

col1 = []
col2 = []

with open(sys.argv[1],'r') as f:
    num_nodes , num_edges = map(int, f.readline().split())
    for l in f:
        n1,n2 = map(int, l.split())
        col1.append(n1)
        col2.append(n2)

dat = [1] * num_edges

csr = sparse.csr_matrix((dat,(col1,col2)),shape=(num_nodes,num_nodes))

with open(sys.argv[2],'w') as f:
    f.write(str(len(csr.indptr)-1) + ' ' + str(len(csr.indices)) + '\n')
    np.savetxt(f,csr.indptr[None,:], fmt='%d', delimiter=' ')
    f.write('\n')
    np.savetxt(f,csr.indices[None,:], fmt='%d', delimiter=' ')
