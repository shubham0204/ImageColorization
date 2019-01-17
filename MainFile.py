
from Parser import Parser
from Colorizer import Model
from skimage.io import imsave
import numpy as np

parser = Parser( 64 )
model = Model()

X = parser.prepare_images_from_dir( 'train_images/' , 'grayscale' )
Y = parser.prepare_images_from_dir( 'train_images/' )
test_X = parser.prepare_images_from_dir( 'test_images/' , 'grayscale' )

np.save( 'sample_data/X.npy' , X )
np.save( 'sample_data/Y.npy' , Y )
np.save( 'sample_data/test_X.npy' , test_X )
print( 'data processed' )

model.load_model( 'models/final_model.h5' )

#model.fit( X , Y  , number_of_epochs=100 )
#model.save_model( 'models/model.h5')

values = model.predict( test_X )
values = np.maximum( values , 0 )
for i in range( test_X.shape[0] ):
    image_final = ( values[i] * 255).astype( np.uint8)
    imsave( 'predictions/{}.png'.format( i + 1 ) , image_final  )

