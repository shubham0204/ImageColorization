# ImageColorization
A Convolutional Auto Encoder model made to colorize grayscale images using TensorFlow and Keras.
Auto Encoders are special types of feed-forward neural nets which try to create a copy of the
data which is given to them. They can be used in supervised and unsupervised learning.
---
# About the Project
We are going to create a Convolutional Auto Encoder which could colorize grayscale images of human faces.
You can find the [Google Colab notebook](https://colab.research.google.com/drive/1iuyU7c0pSq4mhbo1zLsIlWS_FdziKlAf) here.
The `model.py` file could be found [here](https://github.com/shubham0204/ImageColorization/blob/master/Colorizer.py)

- The model uses Conv2D layer to minimize the image to store it as an encoded image.
- The Conv2DTranspose layers generate the colorized image from the encoded representation of the image.

# Regarding the notebook
For using the notebook in Google Colab :
1. Select the GPU runtime.
2. Upload all the files under the `sample_data` directory to Google Colab.
3. If you wish to continue training, you can upload the `final_model.h5` file located in the 
`models` directory.

For using the notebook in other ways:
- You need to change the paths from where the `X.npy` , `Y.npy` and `test_X.npy` files are loaded.