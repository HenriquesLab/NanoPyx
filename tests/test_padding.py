import numpy as np
from nanopyx.core.transform.dimensions import padding

def test_pad_2d():

    test_array = np.zeros((512,512), dtype=np.float32)

    # Even crops
    padded_1 = padding(test_array,256,256)
    padded_2 = padding(test_array,0,256)
    padded_3 = padding(test_array,256,0)

    assert padded_1.shape == (test_array.shape[0]+256,test_array.shape[1]+256)
    assert padded_2.shape == (test_array.shape[0],test_array.shape[1]+256)
    assert padded_3.shape == (test_array.shape[0]+256,test_array.shape[1])
    
    assert padded_1.sum() == test_array.sum()
    assert padded_2.sum() == test_array.sum()
    assert padded_3.sum() == test_array.sum()

    # Odd crops
    padded_4 = padding(test_array,255,255)
    padded_5 = padding(test_array,0,255)
    padded_6 = padding(test_array,255,0)

    assert padded_4.shape == (test_array.shape[0]+255,test_array.shape[1]+255)
    assert padded_5.shape == (test_array.shape[0],test_array.shape[1]+255)
    assert padded_6.shape == (test_array.shape[0]+255,test_array.shape[1])
       
    assert padded_4.sum() == test_array.sum()
    assert padded_5.sum() == test_array.sum()
    assert padded_6.sum() == test_array.sum()
    

def test_pad_3d():

    test_array = np.zeros((512,512,512), dtype=np.float32)

    # Even crops
    padded_1 = padding(test_array,256,256)
    padded_2 = padding(test_array,0,256)
    padded_3 = padding(test_array,256,0)

    assert padded_1.shape == (test_array.shape[0],test_array.shape[1]+256,test_array.shape[2]+256)
    assert padded_2.shape == (test_array.shape[0],test_array.shape[1],test_array.shape[2]+256)
    assert padded_3.shape == (test_array.shape[0],test_array.shape[1]+256,test_array.shape[2])
    
    assert padded_1.sum() == test_array.sum()
    assert padded_2.sum() == test_array.sum()
    assert padded_3.sum() == test_array.sum()

    # Odd crops
    padded_4 = padding(test_array,255,255)
    padded_5 = padding(test_array,0,255)
    padded_6 = padding(test_array,255,0)

    assert padded_4.shape == (test_array.shape[0],test_array.shape[1]+255,test_array.shape[2]+255)
    assert padded_5.shape == (test_array.shape[0],test_array.shape[1],test_array.shape[2]+255)
    assert padded_6.shape == (test_array.shape[0],test_array.shape[1]+255,test_array.shape[2])
       
    assert padded_4.sum() == test_array.sum()
    assert padded_5.sum() == test_array.sum()
    assert padded_6.sum() == test_array.sum()