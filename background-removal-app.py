import streamlit as st
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img, load_img
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.applications import mobilenet_v2



chan_dim = -1

if K.image_data_format() == "channels_first":
  input_shape = (depth, height, width)
  chan_dim = 1

class Unet:
  @staticmethod

  def conv_module(x, K):
    x = Conv2D(K, (3,3), padding="same", activation="relu")(x)
    x = BatchNormalization(axis=chan_dim)(x)
    x = Conv2D(K, (3,3), padding="same", activation="relu")(x)
    x = BatchNormalization(axis=chan_dim)(x)

    return x
  
  def conv_upconv_module(x, K):
    x = Conv2D(K, (3,3), padding="same", activation="relu")(x)
    x = BatchNormalization(axis=chan_dim)(x)
    x = Conv2D(K, (3,3), padding="same", activation="relu")(x)
    x = BatchNormalization(axis=chan_dim)(x)
    x = Dropout(0.2)(x)
    x = Conv2DTranspose(K/2, (2,2), strides=(2, 2), padding='same')(x)

    return x

  def build(input_shape):

    input = Input(shape=input_shape)

    c1 = BatchNormalization(axis=chan_dim)(input)
    c1 = Unet.conv_module(c1, 16)
    p1 = MaxPooling2D(pool_size=(2,2))(c1)
    p1 = Dropout(0.2)(p1)
    
    c2 = Unet.conv_module(p1, 32)
    p2 = MaxPooling2D(pool_size=(2,2))(c2)
    p2 = Dropout(0.2)(p2)

    c3 = Unet.conv_module(p2, 64)
    p3 = MaxPooling2D(pool_size=(2,2))(c3)
    p3 = Dropout(0.2)(p3)

    c4 = Unet.conv_module(p3, 128)
    p4 = MaxPooling2D(pool_size=(2,2))(c4)
    p4 = Dropout(0.2)(p4)

    u6 = Unet.conv_upconv_module(p4, 256)
    u6 = concatenate([u6, c4], axis=chan_dim)

    u7 = Unet.conv_upconv_module(u6, 128)
    u7 = concatenate([u7, c3], axis=chan_dim)

    u8 = Unet.conv_upconv_module(u7, 64)
    u8 = concatenate([u8, c2], axis=chan_dim)

    u9 = Unet.conv_upconv_module(u8, 32)
    u9 = concatenate([u9, c1], axis=chan_dim)

    c9 = Conv2D(16, (3,3), padding="same", activation="relu")(u9)
    c9 = BatchNormalization(axis=chan_dim)(c9)
    c9 = Conv2D(16, (3,3), padding="same", activation="relu")(c9)
    c9 = BatchNormalization(axis=chan_dim)(c9)
    c9 = Dropout(0.2)(c9)
    output = Conv2D(1, (1, 1), padding="same", activation='sigmoid')(c9)

    model = Model(inputs=input, outputs=output)

    return model

def get_final_shape(image):
    array_im = np.array(image)
    orig_shape = array_im.shape
    ratio = orig_shape[0] / 224
    final_shape = (224, int(orig_shape[0]/ratio))
		
    return final_shape

def is_correctly_oriented(image): 
    
    # returns the estimated probability that the given image has correct orientation
    # assumes that the image has shape of (224,224,3)
    image = img_to_array(image)[:,:,:3]
    
    # preprocessing function of MobileNetV2
    preprocessed_image = mobilenet_v2.preprocess_input(image)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    
    correctly_oriented = mobilenetv2.predict(preprocessed_image)[0][0]
    
    return correctly_oriented

def correct_orientation(image):
    # returns the correct orientation of the image, assumes that the image has 
    # shape of (224,224,3)
    
    correctly_oriented = is_correctly_oriented(image)
    
    # if it's very likely to be already correctly oriented, return 0
    if correctly_oriented > 0.9:
        return 0
    
    # otherwise, let's compute all the probabilities:
    probabilities = [correctly_oriented]
    for i in range(1,4):
        rotated_image = np.rot90(image, i)
        probabilities.append(is_correctly_oriented(rotated_image))
        
    n_rotations_needed = np.argmax(probabilities)
    
    return np.rot90(image, n_rotations_needed)



def about():
	st.write(
		'''
		
		
		In this project I wanted to build an algorithm that could automatically remove the background from a selfie. For that, I used the U-net 
		architecture and the AISegment.com Matting Human dataset, which consists of approximately 
		34 thousand pictures of mid-upper body selfies, with only one person in each picture, relatively near from the camera and with a high contrast with the background.
		
		Please take into account that the model was trained on mid-upper body selfies, with only one person in the picture,
		relatively near from the camera and with a high contrast with the background.
		
		**If you want to see how I did it, [here](https://github.com/javiergarciamolina/selfie-background-removal) is the repo.**
				
		''')

unet = Unet.build((224,224,3))
unet.load_weights('unet_weights.h5')
mobilenetv2 = load_model("mobilenetv2.h5")

def main():
  st.title("Selfie Background Removal")

  activities = ["App", "About"]
  choice = st.sidebar.selectbox("Pick something:", activities)

  if choice == "App":
	
    st.write("**Please note that it will work best with mid-upper body selfies, ideally with only one person in the picture,  relatively near from the camera and with a high contrast with the background.**")
    st.write("Here's an example of the kind of pictures with wich it works best:")
	
    selfie_mine = load_img("images/selfie_mine.jpeg", target_size=(224,224,3))
    my_selfie_final_shape = get_final_shape(selfie_mine)
    selfie_mine = array_to_img(selfie_mine)
    selfie_mine = selfie_mine.resize(my_selfie_final_shape)
    st.image(selfie_mine)
	
    st.write("You can go to the About section from the sidebar to learn more about it, or click [here](https://github.com/javiergarciamolina/selfie-background-removal) to see the repository.")
            
    image_file = st.file_uploader("Upload selfie", type=['jpeg', 'png', 'jpg', 'webp'])

    if image_file is not None:

      orig_image = Image.open(image_file)
      final_shape = get_final_shape(orig_image)

      image = orig_image.resize((224,224))
      image = np.array(image)
      image = correct_orientation(image)
   
      image = np.array(image) / 255
      image = np.expand_dims(image, axis=0)
      
	
    if st.button("Process"):
      pred = unet.predict(image)[0]
      pred = unet.predict(image)
      mask = 1-((1-image)*pred)
      mask = array_to_img(mask[0])
      mask = mask.resize(final_shape)
      st.image(mask)

  elif choice == "About":
    about()




if __name__ == "__main__":
    main()
