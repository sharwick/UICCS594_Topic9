from mpi4py import MPI
import sys
import numpy as np
import math

comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
alpha = .05
cinit = (0.0,0.0,0.0)

# The following code is taken from: http://code.activestate.com/recipes/52273-colormap-returns-an-rgb-tuple-on-a-0-to-255-scale-/
def floatRgb(mag, cmin, cmax):
       """
       Return a tuple of floats between 0 and 1 for the red, green and
       blue amplitudes.
       """

       try:
              # normalize to [0,1]
              x = float(mag-cmin)/float(cmax-cmin)
       except:
              # cmax = cmin
              x = 0.5
       blue = min((max((4*(0.75-x), 0.)), 1.))
       red  = min((max((4*(x-0.25), 0.)), 1.))
       green= min((max((4*math.fabs(x-0.5)-1., 0.)), 1.))
       return [red, green, blue]

def floatRgb2(mag):
	g = lambda x: floatRgb(x,-5000,5000)
	return g
floatRgbV = np.vectorize(floatRgb)

#if rank==0:
#	sys.stdout.write("floatRgbV([100,200],-5000,5000) = %s\n" % (str(floatRgbV([100,200],-5000,5000))))
#	sys.stdout.write("floatRgb(100,-5000,5000) = %s\n" % (str(floatRgb(100,-5000,5000))))
#	sys.stdout.write("floatRgb2(100) = %s\n" % (str(floatRgb2(100))))
#	sys.stdout.write("floatRgbV([100,200]) = %s\n" % (str(floatRgbV([100,200]))))

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
	dims = []
	#dims.extend(subset[:,:,i].shape)
	dims.extend( (subset.shape[0], subset.shape[1]))
	dims.append(3)
	sys.stdout.write("Rank = %d, Dims = %s\n" % (rank, str(dims)))
	 
	cout = np.array( [cinit for i in range( dims[0]*dims[1] )]).reshape(dims)
	sys.stdout.write("cout dimensions for rank=%d = %s\n" % (rank, str(cout.shape)))
	
	for i in range(zmax-zmin+1):
		#cin = map(floatRgb2, subset[:,:,i])
#		cin = np.array(floatRgbV(np.reshape(subset[:,:,i],(1,500*500)),-5000,5000))
#		cin = np.reshape(cin,(500,500))

		cin = np.array( [cinit for i in range( dims[0]*dims[1] )]).reshape(dims)
		for x in range(dims[0]):
			for y in range(dims[1]):
				cin[x,y] = floatRgb(subset[x,y,i],-5000,5000)

		if i==0:
			sys.stdout.write("cin dimensions for rank=%d = %s, cin[0,0] = %s\n" % (rank, str(cin.shape), str(cin[0,0])))
		cout = [ [[(1-alpha)*x for x in y] for y in z] for z in  cout] + [ [[(alpha)*x for x in y] for y in z] for z in  cin]
	
	sys.stdout.write("Rank=%d, data = %s\n" % (rank, str(cout[0,0,:])))
		
	comm.Send(cout, dest=0, tag=13)

else:
	sys.stdout.write("Calculating Overall Image\n")

	dims = [xSize, ySize, nproc,3]
	allData = np.arange(xSize*ySize*nproc*3,dtype=np.float64).reshape(dims)

	for i in range(nproc):
		d = np.arange(xSize*ySize*3,dtype=np.float64)
		d.reshape((xSize,ySize,3))
		comm.Recv(d, source=i+1, tag=13)
		allData[:,:,i,:] = d

	finalData = allData[:,:,0,:]
	
	for i in range(1,nproc):
		finalData = (1-alpha)*finalData + alpha*allData[:,:,i,:]
		finalData =  [ [[(1-alpha)*x for x in y] for y in z] for z in  finalData] + [ [[(alpha)*x for x in y] for y in z] for z in  allData]
	
	sys.stdout.write("Final results (rank=%d): %s \n" % (rank,str(finalData.shape),str(finalData)))


	# Write file
	f = open('image.ppm','w')
	f.write("P6\n")
	f.write("# image.ppm\n")
	f.write(str(xSize) + ' ' + str(ySize) + '\n')
	f.write('255\n')

	for x in finalData:
		for y in x:
			for z in y:
				f.write(str(round(z*255,0)) + " ")
		f.write('\n')

	f.close()


