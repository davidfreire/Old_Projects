import numpy as np


def run():

	chunk_size=5 # Memory size
	n_chunks=5 # number of elements if data is divided by the memory size (a 5x5 image, there are 5 chunks (rows) of 5 elements (pixels))

	
	np.random.seed(0)
	#Let's say we have 3 grayscale images (1-channel) of 5x5 pixels 
	x = np.random.random((3,n_chunks,chunk_size))
	x = np.round(10*x)
	print(x.shape)
	print(x)



	#Now for the RNN, we need to convert each row of pixel into a single chunk. 
	#Therefore, we would have 5 chunks of 5 values each
	#The next line swaps the 0th dim with the 1st dim, so we get a (5,3,5) 
	# shape for the images, which is 5 chunks of 3 images each of 5 elements 
	x_t = np.transpose(x,(1,0,2))
	print('Transponse')
	print(x_t.shape)
	print(x_t)

	# The weight matrix inside the RNN is 2D. Then, we flatten de dimension to have a 15x5 matrix.
	x_r = np.reshape(x_t,(-1,n_chunks))
	print('Reduced')
	print(x_r.shape)
	print(x_r)

	#Finally, we need to split the entire structure into 5 chunks (5 arrays).
	x_s = np.split(x_r, n_chunks, 0)
	print('Split')
	print(len(x_s))
	print(x_s)
	print('First input to RNN')
	print(x_s[0])






if __name__ == '__main__':
	run()



