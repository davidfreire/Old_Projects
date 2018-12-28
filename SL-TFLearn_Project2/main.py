import tflearn
from tflearn.data_utils  import to_categorical, pad_sequences
from tflearn.datasets import imdb

def run():
	train, test, _ = imdb.load_data(path='imdb.pkl', n_words = 10000, valid_portion = 0.1) 
	#pkl is a byte stream, makes easier to convert to other python objects such as lists
	#only 10000 words from the dataset are considered and 10% of data for validation set

	trainX, trainY = train
	testX, testY = test

	#Data preprocesing...vectorize inputs into numerical vectors
	#Convert each review into a matrix and pad it in order to keep the consistency of inputs dimensionality. 
	#It will pad the sequences with 0s at the end until it reachs the maximun lenght of the sequence (100)
	trainX = pad_sequences(trainX, maxlen=100, value=0.)
	testX = pad_sequences(testX, maxlen=100, value=0.)

	#Convert labels into vectors. These are binary vectors with to classes: 1 positive and 0 negative
	trainY = to_categorical(trainY, nb_classes=2)
	testY = to_categorical(testY, nb_classes=2)


	#Network building
	net = tflearn.input_data([None, 100]) #input layer. 100 because I set the max sequence length to 100.
	net = tflearn.embedding(net, input_dim=10000, output_dim=128) #next layer
	net = tflearn.lstm(net, 128, dropout=0.8)
	net = tflearn.fully_connected(net, 2, activation ='softmax')
	net = tflearn.regression(net, optimizer='adam', learning_rate=0.0001,
		loss='categorical_crossentropy')

	model = tflearn.DNN(net, tensorboard_verbose = 0)
	model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
		batch_size=32)




if __name__ == '__main__':
	run()