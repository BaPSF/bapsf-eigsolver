# hdf5tools holds useful tools for getting eigsolver (and other) data
# into the hdf5 format for storage or importing into other analysis packages

import h5py

def WriteToH5(filename, variable_dict):
	# function to write all variables in the dictionary to an hdf5 file
        # given a dictionary of variable names and the variables,
        # the variables will be stored in the given HDF5 file
        #
        # usage example:
        #
        ## WriteToH5(fname+'.hdf5', {'gammas':gammas,
        ##                   'omegas':omegas,
        ##                   'eigenmodes':eigenmodes,
        ##                   'locations':locations,
        ##                   'm_nums':m_nums,
        ##                   'bfields':bfields})


	if filename[-5:] != '.hdf5':
		filename=filename+'.hdf5'

	h5f=h5py.File(filename, 'w')
	
	var_names=list(variable_dict.keys())
	for var_name in var_names:
		dataset=h5f.create_dataset(var_name, data=variable_dict[var_name])
		
	dt= h5py.special_dtype(vlen=str)
	dataset=h5f.create_dataset('var_names', data=var_names, dtype=dt)

	h5f.close()
