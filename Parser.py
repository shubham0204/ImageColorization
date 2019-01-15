
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage import color
import os

class Parser ( object ) :

    def __init__(self , dimen ):
        self.__DIMEN = dimen

    def __normalize(self , data):
        max = np.max(data)
        min = np.min(data)
        return (data - min) / max - min

    def prepare_images_from_dir(self, dir_path, mode='rgb' ):
        if mode == 'grayscale':
            images = list()
            images_names = os.listdir(dir_path)
            for imageName in images_names:
                full_path = dir_path + imageName

                image = imread( full_path )
                image = color.rgb2gray( image )
                image = resize( image , ( self.__DIMEN , self.__DIMEN , 1 ))
                images.append(  image )

            return self.__normalize(np.array(images))
        elif mode == 'rgb' :
            images = list()
            images_names = os.listdir(dir_path)
            for imageName in images_names:

                full_path = dir_path + imageName
                image = imread(full_path, as_gray=False)
                image = resize(image, (self.__DIMEN, self.__DIMEN, 3))
                images.append(image)

            return self.__normalize(np.array(images))
