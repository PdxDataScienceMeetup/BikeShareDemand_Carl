# Kaggle: Bike sharing demand

My scripts for modeling the bike sharing demand data from Kaggle.
For more info: https://www.kaggle.com/c/bike-sharing-demand

## iPython notebook

Explore the data by installing the latest ipython (and pandas,
matplotlib, numpy) and typing

```
ipython notebook
```

You can plot the rider count vs many input variables, while
restricting the plot to arbitrary slices of input space.

## nonlinear.py

This is my custom brute-force optimization stuff. No scikit-learn here.
My models combine parameters and inputs in various nonlinear ways and just
send the result into scipy optimization routines.

I spent a little time developing a framework that helps the model-builder
return the gradient of a composite model function using sum and product
rules for derivatives. This helps convergence in the scipy optimization
routines.

See `models.py` and several model implementations in `nonlinear.py`.

## svm.py

This uses scikit-learn's support vector regression (SVR) module. As a
black box, it does not perform great. I had to mess with the kernel to get
decent approximations. Highest score on Kaggle was about 0.6, which is
pretty far down the leaderboard.

## License

MIT
