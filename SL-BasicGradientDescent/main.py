from numpy import *

def Compute_Error(data,m,b):
    error=0
    for i in range(len(data)):
        x = data[i,0]
        y = data[i,1]
        error += (y-(m*x + b))**2
    error /= len(data)
    return error

def Gradient_Step(data,m,b):
    grad_m = 0
    grad_b = 0
    for i in range(len(data)):
        x = data[i,0]
        y = data[i,1]

        grad_m += ((y-(m*x + b))*x)
        grad_b += (y-(m*x + b))
    grad_m *= -2/len(data)
    grad_b *= -2/len(data)
    return [grad_m, grad_b]


def Gradient_Descend(num_iter, learning_rate, data, m, b):

    for i in range(num_iter):
        [grad_m, grad_b] = Gradient_Step(data, m, b)
        m += - learning_rate * grad_m
        b += -learning_rate * grad_b

    return [m,b]



if __name__ == '__main__':
    def run():
        #Load data
        data = genfromtxt('data.csv',delimiter=',')
        #print(data)

        #Init hyperparam
        num_iter = 100
        learning_rate =0.0001
        m=0
        b=0

        #Run the algorithm
        print('Old parameters: m=%f   b=%f   error=%f' %(m,b,Compute_Error(data,m,b)))
        [m,b] = Gradient_Descend(num_iter, learning_rate, data, m, b)
        print('Trained parameters: m=%f   b=%f   error=%f' %(m,b,Compute_Error(data,m,b)))

if __name__=='__main__':
    run()
