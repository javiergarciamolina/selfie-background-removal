# Importing required libraries, obviously
import streamlit as st
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img, load_img


# Loading pre-trained parameters for the cascade classifier

unet = load_model("trained_unet.h5")

def threshold_pred(pred, thresh_low=0, thresh_high=0.2):
  thresh_pred = np.zeros(pred.shape)
  for i in range(pred.shape[0]):
    for j in range(pred.shape[1]):
      if pred[i,j] < thresh_low:
        thresh_pred[i,j] = 0

      elif pred[i,j] > thresh_high:
        thresh_pred[i,j] = 1

      else:
        thresh_pred[i,j] = pred[i,j]

  return thresh_pred

def about():
	st.write(
		'''
		I built this app training a U-net. Here are the results.
		''')


def main():
  st.title("Selfie Background Removal")

  activities = ["App", "About"]
  choice = st.sidebar.selectbox("Pick something:", activities)

  if choice == "App":
	
    st.write("**Please note that it will work best with selfies, ideally with only one person in the picture,  relatively near from the camera and with a high contrast with the background.**")
    st.write("Here's an example of the kind of pictures with wich it works best:")
	
    selfie_mine = load_img("selfie_mine.jpeg")
    selfie_mine = array_to_img(selfie_mine)
    st.image(selfie_mine)
	
    st.write("You can go to the About section from the sidebar to learn more about it.")
            
    # You can specify more file types below if you want
    image_file = st.file_uploader("Upload selfie", type=['jpeg', 'png', 'jpg', 'webp'])

    if image_file is not None:

      image = Image.open(image_file)
      image = image.resize((224,224))
   
      image = np.array(image) / 255
      image = np.expand_dims(image, axis=0)
	
    if st.button("Process"):
                    
      # result_img is the image with rectangle drawn on it (in case there are faces detected)
      # result_faces is the array with co-ordinates of bounding box(es)
      pred = unet.predict(image)[0]
      pred = threshold_pred(pred)
      
      #image = img_to_array(image)
      mask = 1-((1-image)*pred)
      mask = array_to_img(mask[0])
      st.image(mask)
	       #,use_column_width=True)

  elif choice == "About":
    about()




if __name__ == "__main__":
    main()