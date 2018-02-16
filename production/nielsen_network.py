## Libaries
import json
import random
import sys
import numpy as np

class QuadraticCost(object):
	@staticmethod
	def fn(a, y):
		"""return the cost associated with an observed output a and a desired output y"""
		return 0.5*np.linalg.norm(a-y)**2
	@staticmethod

	def delta(z, a, y):
		"""return the error delta for a layer"""
		return (a-y)*sigmoid_prime(z)

class CrossEntropyCost(object):
	@staticmethod
	def fn(a,y):
		""""return cost associated with observed output a and desired output y. make sure that nan is 0.0"""
		return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))
	def delta(z, a, y ):
		"""return the error delata of an output layer"""
		return (a-y)



# Main Network class
class nertwork(object):

	def __ini__(self, sizes, cost=CrossEntropyCost):
		"""sizes [1,2,3] is 1 neuron is first layer, 2 in second and 3 in third and so on
		biases and weights are intiallized randomly"""
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.default_weight_initializer()
		self.cost = cost
	def default_weight_initializer(self):
		"""initialize weight using a gaussianj distribution, mu=0, std=1.
		The first layer is the iput later so we dont set any biases for those"""
		self.biases= [np.random(y,1) for y in self.sizes[1:]]
		self.weights= [np.random(y,x)/np.sqrt(x) for x,y in zip(self.sizes[:-1],self.sizes[1:])]

	def feedforward(self, a):
		"""return the output of a network of the input a"""
		for b,w in zip(self.biases, self.weights):
			a = sigmoid(np.dot(w,a) + b)
		return a
	#def SGD(self)
	def save(self, filname):
		"""save nn to a file"""
		data = {"sizes": self.sizes, "weights":[w.tolist() for w in self.weights], "biases": [b.tolist() for b in self.biases], "cost": str(self.cost.__name__)}
		f = open(filename, "r")
		json.dump(data,f)
		f.close()


