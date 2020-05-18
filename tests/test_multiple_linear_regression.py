from .context import scratchml
from scratchml.regression import MultipleLinearRegression
import numpy as np

def test_initialization():
    mr = MultipleLinearRegression()
    assert isinstance(mr, MultipleLinearRegression)

def test_simple_fit_and_predict():
    mr = MultipleLinearRegression()
    coef = np.random.randn(2,1)

    x = np.array(np.random.randn(int(1e7), 1))
    y = coef[0] + x * coef[1]
    yhat = mr.fit(x, y).predict(x)

    # Checking values with isclose because of float precision issues in python
    assert np.isclose(mr.coef_, coef).all()
    assert np.isclose(y, yhat).all()

def test_multple_fit_and_predict():
    mr = MultipleLinearRegression()

    n_features = np.random.randint(50)
    n_samples = np.random.randint(3 * int(1e4))

    weights = np.random.randn(n_features,)
    bias = np.random.randint(-1000, 1000)

    x = np.random.randn(n_samples, n_features)
    y = bias + np.sum((weights * x), axis=1)
    
    yhat = mr.fit(x, y).predict(x)

    assert np.isclose(mr.coef_, np.concatenate((np.array([bias]), weights))).all()
    assert np.isclose(y, yhat).all()

def test_evaluation():
    mr = MultipleLinearRegression()

    n_features = np.random.randint(50)
    n_samples = np.random.randint(3 * int(1e4))

    weights = np.random.randn(n_features,)
    bias = np.random.randint(-1000, 1000)

    x = np.random.randn(n_samples, n_features)
    y = bias + np.sum((weights * x), axis=1)
    
    mr.fit(x, y)

    assert mr.evaluate(x, y)['score'] == 1