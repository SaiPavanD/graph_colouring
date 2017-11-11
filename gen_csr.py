import numpy as np
from scipy import sparse
import sys

if(len(sys.argv)!=3):
    print 'Usage : python gen_csr <ip_file> <op_file>'
    sys.exit(0)

with open(sys.argv[1],'r') as f:
    n = int(f.readline())
    arr = np.zeros(shape=(n,n))
    for l in f:
        n1,n2 = map(int, l.split(' '))
        if n1 == n2:
            continue
        arr[n1][n2] = 1
        arr[n2][n1] = 1

csr = sparse.csr_matrix(arr)

with open(sys.argv[2],'w') as f:
    f.write(str(len(csr.indptr)) + ' ' + str(len(csr.indices)) + '\n')
    np.savetxt(f,csr.indptr[None,:], fmt='%d', delimiter=' ')
    np.savetxt(f,csr.indices[None,:], fmt='%d', delimiter=' ')    
