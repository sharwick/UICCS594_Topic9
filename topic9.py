from mpi4py import MPI
import sys
import numpy as np
import math

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

#sys.stdout.write("Helloworld! I am process %d of %d on %s.\n" % (rank, size, name))

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

	xSize = math.floor(xN/xpartitions)
	ySize = math.floor(yN/ypartitions)
	zSize = math.floor(zN/zpartitions)

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

	
				#comm.Send([[xmin,xmax,ymin,ymax,zmin,zmax],MPI.INT],dest=proc,tag=77)
				toSend = np.array([xmin,xmax,ymin,ymax,zmin,zmax])
				comm.Send([toSend,MPI.INT],dest=proc,tag=77)
# 2 + 2*3 + 2*3*3 = 2 + 6 + 18 = `26

else:
	comm.Recv([data,MPI.INT],dest=proc,tag=77)
	[xmin,xmax,ymin,ymax,zmin,zmax] = np.asarray(data)
	sys.stdout.write("Process %d received %s\n" %(rank, [xmin,xmax,ymin,ymax,zmin,zmax]))

comm.Barrier()
data = comm.bcast(data, root=0)
comm.Barrier()


sys.stdout.write("Process %d: %s\n" % (rank, str(data.shape())))
