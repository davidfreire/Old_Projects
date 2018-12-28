import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import NN as nn


def bike_data_preprocessing(dataset):
	print('Load dataset: ')
	print(dataset.head())


	#-----------DUMMY VARIABLES---------
	#There are categorical variables such as month, season and weather.
	#To include these in our model, we need to make them binary dummy variables
	#We do it with panda:

	dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
	for each in dummy_fields:
		#create the dummy matrix
		dummies = pd.get_dummies(dataset[each], prefix=each, drop_first=False)
		#concatenate the dummy matrix
		dataset = pd.concat([dataset, dummies], axis=1)
	
	#Now we can drop irrelevant variables and categorical variables (dummy variable now appear as i.e. hr_1 hr_2 hr_3 etc.)
	fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 'mnth', 'hr', 'weekday', 'workingday', 'atemp']

	data = dataset.drop(fields_to_drop, axis=1)
	print('New dataset:')
	print(data.head())

	#-----------SCALE VARIABLES---------
	#Makes the training process much easier. 
	#The idea is data to have 0 mean and standard deviation of 1
	#Dummy variables are already 0 and 1, then:
	quant_features = ['casual', 'temp', 'hum', 'windspeed', 'registered', 'cnt']
	#variables such as 'cnt', 'casual', 'registered' are targets, and they are going to be removed.
	scaled_features = {}
	for each in quant_features:
		mean,std = data[each].mean(), data[each].std()
		scaled_features[each] = [mean, std]
		data.loc[:,each] = (data[each] - mean) /std


	print('Dataset:')
	print(data.head())

	return data


def MSE(y, Y):
    return np.mean((y-Y)**2)




def run():
	#Get data
	data_path = 'dataset/hour.csv'

	dataset = pd.read_csv(data_path)

	dataset = bike_data_preprocessing(dataset=dataset)

	print("Dataset shape")
	print(dataset.shape)

	#Split data intro training, testing and validation sets.
	#test with the last 21 days
	test_data = dataset[-21*24:] #there are 24 observations (lines) per day
	#the rest for train a validation
	data = dataset[:-21*24]

	#remove the data into features and targets (in this example we will just consider cnt but the others can be used as well)
	target_fields = ['cnt', 'casual', 'registered']
	features, targets = data.drop(target_fields, axis=1), data[target_fields]

	#remove it from the test data too
	test_features, test_targets = data.drop(target_fields, axis=1), data[target_fields]

	#split data into trainign a validation. Las 60 days of the remaining data as a validation set
	train_features, train_targets = features[:-60*24], targets[:-60*24]
	validation_features, validation_targets = features[-60*24:], targets[-60*24:]


	#Set hyperparameters
	epochs = 500 #Must choose enough epochs to train the NN but not too many to overfit it.
	learning_rate = 0.1 #Scales the size of the weight updates. A good choice to start at is 0.1. If the network has problems fitting the data, try reducing the learning rate. Note that the lower the learning rate, the smaller the steps are in the weight updates and the longer it takes for the neural network to converge.
	hidden_nodes = 27 #The more hidden nodes you have, the more accurate predictions the model will make.  If the number of hidden units is too low, then the model won't have enough space to learn and if it is too high there are too many options for the direction that the learning can take. The trick here is to find the right balance in number of hidden units you choose.
	output_nodes = 1 


	#Create the neural network
	num_inputs = train_features.shape[1]
	network = nn.NeuralNetwork(num_inputs, hidden_nodes, output_nodes, learning_rate)



	#Training by using SGD.
	losses = {'train':[], 'validation':[]}
	
	for e in range(epochs):
		#Go through a random batch of 128 records of the training dataset
		batch = np.random.choice(train_features.index, size=128)
		for record,target in zip(train_features.ix[batch].values, train_targets.ix[batch]['cnt'].values):
			#Train the network with each batch sample
			network.train(record, target)

		#Print out the training process
		train_loss = MSE(network.test(train_features), train_targets['cnt'].values)

		validation_loss = MSE(network.test(validation_features), validation_targets['cnt'].values)

		sys.stdout.write("\rProgress: " + str(100 * e/float(epochs))[:4] \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(validation_loss)[:5])


		losses['train'].append(train_loss)
		losses['validation'].append(validation_loss)

	

	















if __name__ == '__main__':
	run()


