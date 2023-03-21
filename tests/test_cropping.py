import numpy as np
from nanopyx.core.transform.dimensions import crop


def test_crop_2d():

    test_array = np.eye(512)

    # Even crops
    cropped_1 = crop(test_array,256,256)
    cropped_2 = crop(test_array,0,256)
    cropped_3 = crop(test_array,256,0)

    assert cropped_1.shape == (test_array.shape[0]-256,test_array.shape[1]-256)
    assert cropped_2.shape == (test_array.shape[0],test_array.shape[1]-256)
    assert cropped_3.shape == (test_array.shape[0]-256,test_array.shape[1])
    
    assert cropped_1.sum() == np.eye(test_array.shape[0]-256,test_array.shape[1]-256).sum()
    assert cropped_2.sum() == np.eye(test_array.shape[0],test_array.shape[1]-256).sum()
    assert cropped_3.sum() == np.eye(test_array.shape[0]-256,test_array.shape[1]).sum()

    # Odd crops
    cropped_4 = crop(test_array,255,255)
    cropped_5 = crop(test_array,0,255)
    cropped_6 = crop(test_array,255,0)

    assert cropped_4.shape == (test_array.shape[0]-255,test_array.shape[1]-255)
    assert cropped_5.shape == (test_array.shape[0],test_array.shape[1]-255)
    assert cropped_6.shape == (test_array.shape[0]-255,test_array.shape[1])
       
    assert cropped_4.sum() == np.eye(test_array.shape[0]-255,test_array.shape[1]-255).sum()
    assert cropped_5.sum() == np.eye(test_array.shape[0],test_array.shape[1]-255).sum()
    assert cropped_6.sum() == np.eye(test_array.shape[0]-255,test_array.shape[1]).sum()
    

def test_crop_3d():

    eye = np.eye(512,512)
    test_array = np.repeat(eye[np.newaxis,:, :], 3, axis=0)

    # Even crops
    cropped_1 = crop(test_array,256,256)
    cropped_2 = crop(test_array,0,256)
    cropped_3 = crop(test_array,256,0)

    assert cropped_1.shape == (test_array.shape[0],test_array.shape[1]-256,test_array.shape[2]-256)
    assert cropped_2.shape == (test_array.shape[0],test_array.shape[1],test_array.shape[2]-256)
    assert cropped_3.shape == (test_array.shape[0],test_array.shape[1]-256,test_array.shape[2])

    assert cropped_1.sum() == np.eye(test_array.shape[1]-256,test_array.shape[2]-256).sum() * test_array.shape[0]
    assert cropped_2.sum() == np.eye(test_array.shape[1],test_array.shape[2]-256).sum() * test_array.shape[0]
    assert cropped_3.sum() == np.eye(test_array.shape[1]-256,test_array.shape[2]).sum() * test_array.shape[0]


    # Odd crops
    cropped_4 = crop(test_array,255,255)
    cropped_5 = crop(test_array,0,255)
    cropped_6 = crop(test_array,255,0)

    assert cropped_4.shape == (test_array.shape[0],test_array.shape[1]-255,test_array.shape[2]-255)
    assert cropped_5.shape == (test_array.shape[0],test_array.shape[1],test_array.shape[2]-255)
    assert cropped_6.shape == (test_array.shape[0],test_array.shape[1]-255,test_array.shape[2])
    
    assert cropped_4.sum() == np.eye(test_array.shape[1]-255,test_array.shape[2]-255).sum() * test_array.shape[0]
    assert cropped_5.sum() == np.eye(test_array.shape[1],test_array.shape[2]-255).sum() * test_array.shape[0]
    assert cropped_6.sum() == np.eye(test_array.shape[1]-255,test_array.shape[2]).sum() * test_array.shape[0]
    