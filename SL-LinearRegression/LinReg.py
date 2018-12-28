import pandas as pd
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt




def run():
	#read the data
	data = pd.read_fwf('brain_body.txt')
	#print(data)
	x = data[['Brain']]
	y = data[['Body']]
	print(type(x))
	diabetes = datasets.load_diabetes()
	print(type(diabetes))

	#train model
	model = linear_model.LinearRegression()
	model.fit(x,y)

	#visualize results
	plt.scatter(x,y)
	plt.plot(x, model.predict(x))
	plt.show()

	#example
	print(model.predict([[27],[248]]))



if __name__ == '__main__':
	run()
