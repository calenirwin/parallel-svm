import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    test = np.arange(0,512,dtype='float64')
    test = np.tile(test,[256,1]) #Create 2D input array. Numbers 1 to 512 increment across dimension 2.
    split = np.array_split(test,size,axis = 0) #Split input array by the number of available cores
    print(np.shape(split))
    split_sizes = []

    for i in range(0,len(split),1):
        split_sizes = np.append(split_sizes, len(split[i]))

    split_sizes_input = split_sizes*512
    displacements_input = np.insert(np.cumsum(split_sizes_input),0,0)[0:-1]


    print(test)
    print("Input data split into vectors of sizes %s" %split_sizes_input)
    print("Input data split with displacements of %s" %displacements_input)

    

else:
#Create variables on other cores
    split_sizes_input = None
    displacements_input = None
    split = None
    test = None
    outputData = None

split = comm.bcast(split, root=0) #Broadcast split array to other cores

output_chunk = np.zeros(np.shape(split[rank])) #Create array to receive subset of data on each core, where rank specifies the core
print("Rank %d with output_chunk shape %s" %(rank,output_chunk.shape))
comm.Scatterv([test,split_sizes_input, displacements_input,MPI.DOUBLE],output_chunk,root=0)
print(output_chunk.shape)
print(output_chunk)

output = np.zeros([len(output_chunk),512]) #Create output array on each core

for i in range(0,np.shape(output_chunk)[0],1):
    output[i,0:512] = output_chunk[i]

print("Output shape %s for rank %d" %(output.shape,rank))


comm.Barrier()
