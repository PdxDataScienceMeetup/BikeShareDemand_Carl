import numpy
from matplotlib import pyplot as plt

class Model:
  def __init__(self, train):
    self._train = train

  def fit(self, beta, grad=True):
    return self(self._train, beta, grad=grad)


class MultiModel(Model):
  def __init__(self, train, terms):
    """
    Arguments:
      train {*} - The training data for the model.
      terms {list} - A list of models.
    """
    Model.__init__(self, train)
    self.terms = terms
    #self.terms = [term(train) for term in terms]
    self._beta0 = []
    self._bounds = []
    self.slices = []

    i = 0

    for term in self.terms:
      n = len(term.beta0())
      self._beta0 += term.beta0()
      self._bounds += term.bounds()
      self.slices.append(slice(i, i + n))
      i += n

  def bounds(self):
    return self._bounds

  def beta0(self):
    return self._beta0

  def show(self, beta):
    for term, slc in zip(self.terms, self.slices):
      term.show(beta[slc])


class ProductModel(MultiModel):
  def __init__(self, data, terms=[]):
    """
    Arguments:
      data {*} - The training data for the model.
      terms {list} - A list of Model classes.
    """
    MultiModel.__init__(self, data, terms)

  def __call__(self, x, beta, grad=False):
    N = len(x)
    prod = lambda x, y: x * y

    values = [term(x, beta[slc], grad=grad) for (term, slc) in zip(self.terms, self.slices)]

    if grad == False:
      return reduce(prod, values)

    fs, grads = zip(*values)

    # Work on gradient
    grad = numpy.zeros((N, len(beta)))
    ones = numpy.ones(N)

    for i, (term, slc) in enumerate(zip(self.terms, self.slices)):
      coeffs = reduce(prod, fs[:i], ones) * reduce(prod, fs[i + 1:], ones)
      grad[:, slc] = coeffs.reshape(N, 1) * grads[i]

    return reduce(prod, fs), grad


class SumModel(MultiModel):
  def __init__(self, data, terms=[]):
    """
    Arguments:
      data {*} - The training data for the model.
      terms {list} - A list of Model classes.
    """
    MultiModel.__init__(self, data, terms)

  def __call__(self, x, beta, grad=False):
    summ = lambda x, y: x + y
    values = [term(x, beta[slc], grad=grad) for (term, slc) in zip(self.terms, self.slices)]

    if grad == False:
      return reduce(summ, values)

    fs, grads = zip(*values)
    return reduce(summ, fs), numpy.hstack(grads)


class AsymmGaussian(Model):
  def __init__(self, train, lo=True, hi=True, name='Asymm Gaussian'):
    Model.__init__(self, train)
    self.lo = lo
    self.hi = hi
    self.name = name

  def _unpack_beta(self, beta):
    if self.lo == False:
      return [beta[0], None, beta[1]]
    elif self.hi == False:
      return beta + [None]
    else:
      return beta

  def bounds(self):
    b = [(None, None)]
    widths = 2
    if self.lo == False or self.hi == False:
      widths = 1
    return b + widths * [(0.0, None)]

  def show(self, beta):
    print '%s' % self.name
    print '  center: %.1f' % beta[0]
    if self.lo:
      print '  left width:   %.1f' % beta[1]
    if self.hi:
      if self.lo:
        width = beta[2]
      else:
        width = beta[1]
      print '  right width:  %.1f' % width
    print '  params:', beta

    #plt.scatter(self._train, self(self._train, beta, grad=False))
    #plt.show()
    #plt.clf()

  def __call__(self, t, beta, grad=True):
    f = numpy.ones(len(t))
    mid, dt_lo, dt_hi = self._unpack_beta(beta)
    numer = -0.5 * (t - mid)**2

    hi = t >= mid
    lo = t < mid

    if (self.lo):
      f[lo] = numpy.exp(numer[lo] / dt_lo**2)
    if (self.hi):
      f[hi] = numpy.exp(numer[hi] / dt_hi**2)

    if grad == False:
      return f

    grad = numpy.zeros((len(t), len(beta)))
    diff = (t - mid)
    diff2 = diff**2

    if self.lo:
      # d f / d mid
      grad[lo, 0] = diff[lo] * f[lo] / dt_lo**2
      # d f / d dt_lo
      grad[lo, 1] = diff2[lo] * f[lo] / dt_lo**3

    if self.hi:
      if self.lo:
        index = 2
      else:
        index = 1

      # d f / d mid
      grad[hi, 0] = diff[hi] * f[hi] / dt_hi**2
      # d f / d dt_hi
      grad[hi, index] = diff2[hi] * f[hi] / dt_hi**3

    return f, grad
