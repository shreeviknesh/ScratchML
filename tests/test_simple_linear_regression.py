from .context import scratchml
from scratchml.regression import SimpleLinearRegression
import numpy as np

def test_initialization():
    sr = SimpleLinearRegression()
    assert isinstance(sr, SimpleLinearRegression)

def test_simple_fit_and_predict():
    sr = SimpleLinearRegression()
    coef = np.random.randn(2,)

    x = np.arange(1, 10000)
    y = coef[0] + x * coef[1]
    yhat = sr.fit(x, y).predict(x)

    # Checking values with isclose because of float precision issues in python
    assert np.isclose(sr.coef_, coef).all()
    assert np.isclose(y, yhat).all()

def test_some_random_values_fit_and_predict():
    sr = SimpleLinearRegression()
    coef = np.random.randn(2,)

    x = np.array(np.random.randn(1000, 1))
    y = coef[0] + x * coef[1]
    yhat = sr.fit(x, y).predict(x)

    assert np.isclose(sr.coef_, coef).all()
    assert np.isclose(y, yhat).all()

def test_a_lot_of_random_values_fit_and_predict():
    sr = SimpleLinearRegression()
    coef = np.random.randn(2,)

    x = np.array(np.random.randn(int(1e7), 1))
    y = coef[0] + x * coef[1]
    yhat = sr.fit(x, y).predict(x)

    assert np.isclose(sr.coef_, coef).all()
    assert np.isclose(y, yhat).all()

def test_evaluation():
    sr = SimpleLinearRegression()
    coef = np.random.randn(2,)

    x = np.array(np.random.randn(1000, 1))
    y = coef[0] + x * coef[1]
    sr.fit(x,y)

    assert sr.evaluate(x, y)['score'] == 1