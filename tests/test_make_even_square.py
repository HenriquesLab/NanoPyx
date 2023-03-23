import numpy as np
from nanopyx.core.transform.dimensions import make_even_square

def test_mes_2D():

    test_array_1 = np.zeros((10,10), dtype=np.float32)
    test_array_2 = np.zeros((11,10), dtype=np.float32)
    test_array_3 = np.zeros((11,11), dtype=np.float32)
    test_array_4 = np.zeros((10,11), dtype=np.float32)

    squared_1 = make_even_square(test_array_1)
    squared_2 = make_even_square(test_array_2)
    squared_3 = make_even_square(test_array_3)
    squared_4 = make_even_square(test_array_4)

    assert squared_1.shape == (10,10)
    assert squared_2.shape == (10,10)
    assert squared_3.shape == (10,10)
    assert squared_4.shape == (10,10)

def test_mes_3D():

    test_array_1 = np.zeros((10,10,10), dtype=np.float32)
    test_array_2 = np.zeros((10,11,10), dtype=np.float32)
    test_array_3 = np.zeros((10,11,11), dtype=np.float32)
    test_array_4 = np.zeros((10,10,11), dtype=np.float32)

    squared_1 = make_even_square(test_array_1)
    squared_2 = make_even_square(test_array_2)
    squared_3 = make_even_square(test_array_3)
    squared_4 = make_even_square(test_array_4)

    assert squared_1.shape == (10,10,10)
    assert squared_2.shape == (10,10,10)
    assert squared_3.shape == (10,10,10)
    assert squared_4.shape == (10,10,10)

    