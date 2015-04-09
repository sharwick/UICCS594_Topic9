from mpi4py import MPI
import sys
from netCDF4 import Dataset
import numpy as np
import math

size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()

#sys.stdout.write("Helloworld! I am process %d of %d on %s.\n" % (rank, size, name))

if rank == 0:
	if len(sys.argv) != 5:
		sys.stdout.write("Error: Incorrect number of arguments.\n")

	datafile=sys.argv[1]
	xpartitions=sys.argv[2]
	ypartitions=sys.argv[3]
	zpartitions=sys.argv[4]

	nproc = xpartitions*ypartitions*zpartitions

	sys.stdout.write("(datafile, xpartitions, ypartitions, zpartitions) = (%s, %s, %s, %s).\n" % (datafile, xpartitions, ypartitions, zpartitions))

	sys.stdout.write("nproc = %d\n" % (nproc))


	sys.stdout.write("Reading data\n")
	data = Dataset(datafile, 'r', format='NETCDF4')

	xN = len(data.variables['x'])
	yN = len(data.variables['y'])
	zN = len(data.variables['z'])

	xSize = math.floor(xN/xpartitions)
	ySize = math.floor(yN/ypartitions)
	zSize = math.floor(zN/zpartitions)

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
# 2 + 2*3 + 2*3*3 = 2 + 6 + 18 = 26





