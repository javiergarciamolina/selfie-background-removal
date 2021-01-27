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


def main():
  st.title("Selfie Background Removal")

  activities = ["App", "About"]
  choice = st.sidebar.selectbox("Pick something:", activities)

  if choice == "App":
	
    st.write("**Please note that it will work best with mid-upper body selfies, ideally with only one person in the picture,  relatively near from the camera and with a high contrast with the background.**")
    st.write("Here's an example of the kind of pictures with wich it works best:")
	
    selfie_mine = load_img("images/selfie_mine.jpeg", target_size=(224,224,3))
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
      
      mask = 1-((1-image)*pred)
      mask = array_to_img(mask[0])
      st.image(mask)
	       #,use_column_width=True)

  elif choice == "About":
    about()




if __name__ == "__main__":
    main()
