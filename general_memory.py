import numpy as np 

class layer():
	def __init__(self, names=None, dim=None):
		self.all_dims = []
		self.all_names = []
		self.all_mems =  []
		if type(names) != type([]) or type(dim) != type([]):
			self.all_dims += dim
			self.all_names += names
			self.all_mems = [self.mem(i) for i in dim]
		else:
			print("WARNING :: Arguments not given or are not list")

	def mem(self, x):
		return np.prod(x)

	def __add__(self, y):
		l = layer()
		l.all_dims = self.all_dims + y.all_dims
		l.all_names = self.all_names + y.all_names
		l.all_mems = self.all_mems + y.all_mems
		return l

	def __iadd__(self, y):
		self.all_dims += y.all_dims
		self.all_names += y.all_names
		self.all_mems += y.all_mems


class model():
	def __init__(self, input_dim=(), output_dim=(), chans = []):
		self.layers = layer()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.chans = chans

	def __iadd__(self, y):
		self.layers += y




