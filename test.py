from cost_func import LinearOfNonlinear
import numpy

class LinearFunc:
  def __init__(self, beta0=[1.0, 0.0]):
    self.beta0 = beta0

  def set_data(self, data):
    self.data = data

  def __call__(self, beta):
    return beta[0] * self.data + beta[1]

  def grad(self, beta):
    grad = numpy.ones((len(self.data), len(beta)))
    grad[:, 0] = self.data
    grad[:, 1] = 1.0
    return grad


a1 = LinearFunc()
s1 = LinearFunc()
s2 = LinearFunc()

data = numpy.arange(0.0, 1.0, 0.1)

lon = LinearOfNonlinear(data, [a1], [[s1], [s2]])
print lon([1.0, 0.0] * 3)
