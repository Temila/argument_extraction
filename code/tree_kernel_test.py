import tree_kernels 
import tree 

l=0.1 # kernel parameter 
dat = tree.Dataset() 
dat.loadFromFilePrologFormat("inputdataset.txt")
k = tree_kernels.KernelSST(l)
k.printKernelMatrix(dat)