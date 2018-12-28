# Inspired by Siraj Raval - https://www.youtube.com/watch?v=ftMq5ps503w

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm, time



def run():

	#Step 1 - Load de Data
	X_train, Y_train, X_test, Y_test = lstm.load_data('sp500.csv', 50, True)




	#Step 2 - Build Model
	#It is going to be a stack of layers
	model = Sequential()
	#Add our first layer, a LSTM layer.
	model.add(LSTM(input_dim=1, output_dim=50, return_sequences=True)) #50 units in this layer, setting return_sequences to true means this layer output always fed into the next layer
	model.add(Dropout(0.2))
	#Add our second layer, a LSTM with 100 units and return sequence to False which means its output is only fed into the next layer AT THE END of the sequence. It doesnt output a prediction for the sequence but a prediction vector.
	model.add(LSTM(100, return_sequences=False))
	model.add(Dropout(0.2))
	#Add a linear dense layer to aggregate the data from the prediction vector into one single value
	model.add(Dense(output_dim=1))
	model.add(Activation('linear'))

	start = time.time()
	model.compile(loss='mse', optimizer='rmsprop')
	print('Compilation time:', time.time()-start)




	#Step 3 - Train de model
	model.fit(X_train, Y_train, batch_size=512, nb_epoch=1, validation_split=0.05)

	


	#Step 4 - Plot predictions
	predictions = lstm.predict_sequences_multiple(model, X_test, 50, 50)
	lstm.plot_results_multiple(predictions, Y_test, 50)




	print('End...')



if __name__ == '__main__':
	run()