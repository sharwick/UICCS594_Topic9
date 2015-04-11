from mpi4py import MPI
import sys
import numpy as np
import math

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

# Process 0 divides up the data and sends bounds to each other node
if rank == 0:
	if len(sys.argv) != 5:
		sys.stdout.write("Error: Incorrect number of arguments.\n")

	datafile=sys.argv[1]
	xpartitions=int(sys.argv[2])
	ypartitions=int(sys.argv[3])
	zpartitions=int(sys.argv[4])

	nproc = xpartitions*ypartitions*zpartitions

	sys.stdout.write("(datafile, xpartitions, ypartitions, zpartitions) = (%s, %d, %d, %d).\n" % (datafile, xpartitions, ypartitions, zpartitions))
	sys.stdout.write("nproc = %d\n" % (nproc))
	sys.stdout.write("Reading data\n")


	# Read in the dimensions (from the 12B header)
	fint = open(datafile,"r")
	dataInt = np.fromfile(fint, dtype=np.uint32)
	[xN,yN,zN] = dataInt[0:3]

	# Read in the array of data (everything after header)
	ffloat = open(datafile,"r")
	dataFloat = np.fromfile(ffloat, dtype=np.float32)
	data = dataFloat[3:].reshape(xN,yN,zN)

	xSize = int(math.floor(xN/xpartitions))
	ySize = int(math.floor(yN/ypartitions))
	zSize = int(math.floor(zN/zpartitions))

	sys.stdout.write("Shape of data = %s\n" % (str(data.shape)))

	# Assign blocks

	for z in range(zpartitions):
		zmin = z*zSize
		zmax = min((z+1)*zSize - 1,zN)
		for y in range(ypartitions):
			ymin = y*ySize
                	ymax = min((y+1)*ySize - 1,yN)
			for x in range(xpartitions):
				xmin = x*xSize
                		xmax = min((x+1)*xSize - 1,xN)

				proc = 1+ z*ypartitions*xpartitions + y*xpartitions + x
				sys.stdout.write("Subvolume <%d,%d> <%d,%d> <%d,%d> is assigned to process %d\n" % (xmin,xmax,ymin,ymax,zmin,zmax,proc))				
				toSend = np.array([xmin,xmax,ymin,ymax,zmin,zmax],dtype=np.int32)
				comm.Send([toSend,MPI.INT],dest=proc,tag=77)

				sys.stdout.write("Sent: %s\n" % (str(toSend)))


# Individual nodes receive their data bounds
else:
	data = np.arange(6, dtype='i')
	comm.Recv([data,MPI.INT],source=0,tag=77)
	[xmin,xmax,ymin,ymax,zmin,zmax] = np.asarray(data)
	sys.stdout.write("Process %d received %s\n" %(rank, str([xmin,xmax,ymin,ymax,zmin,zmax])))
	subset = np.empty([xmax-xmin+1,ymax-ymin+1,zmax-zmin+1]) # Initialize subset matrix, to be filled in by slice later
	[xN,yN,zN] = [0,0,0] # initialize

sys.stdout.write("Process %d: %s\n\n" % (rank, str(data.shape)))

# Broadcast data.  Each node keeps only the data assigned to it.

#for z in range(max(zmax,zN)):
for z in range(100):
	if rank==0:
		if z==0:
			sys.stdout.write("(zmin, zmax,zN) = (%d, %d,%d)\n" %(zmin, zmax, zN))
		sys.stdout.write("Broadcasting slice = %d\n" %(z))
		comm.Barrier()
		data[:,:,z] = comm.bcast(data[:,:,z], root=0)
		comm.Barrier()
	else:
		comm.Barrier()
		slice = comm.bcast(slice,root=0)
		comm.Barrier()

	if ( (rank != 0) and (z>=zmin) and (z<=zmax) ):
		subset[:,:,z-zmin] = slice[xmin:xmax+1, ymin:ymax+1]
		sys.stdout.write("Process %d received slice %d\n" %(rank, z))


# Each node calculates its mean and reports to node 0, which then calculates total mean.
if rank!=0:	 
	mean = np.mean(subset)	 
	sys.stdout.write("Process %d has data <%d, %d> <%d, %d> <%d, %d> , mean = %f\n" % (rank,xmin, xmax, ymin, ymax, zmin, zmax, mean)) 
	comm.Send(mean, dest=0, tag=13)

else:
	sys.stdout.write("Calculating Overall Mean\n")
	mean = np.arange(nproc,dtype=np.float64)
	for i in range(nproc):
		d = np.arange(1,dtype=np.float64)
		comm.Recv(d, source=i+1, tag=13)
		mean[i] = d[0]
		
	meanOverall = np.mean(mean)
	
	sys.stdout.write("Process 0 receives local means %s and the overall mean = %f\n" % (str(mean),meanOverall))

# Final answer should be: 372.072

