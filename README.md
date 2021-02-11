# Selfie Background Removal: Project Overview
**Created an app that automatically removes the background from a selfie image.**

![ezgif com-gif-maker (2)](https://user-images.githubusercontent.com/70718425/107150769-d453bf80-695f-11eb-967c-6b089e5d8c84.gif)


## App:

You can try it at:

https://share.streamlit.io/javiergarciamolina/selfie-background-removal-app/main/app.py


Please take into account that the model was trained on mid-upper body selfies, with only one person in the picture and relatively near from the camera.

## Steps:

* Downloaded the AISegment.com Matting Human dataset, which consists of approximately 34 thousand pictures of mid-upper body selfies and their respective masks.
* Tried a couple U-net-like architectures and selected the best, achieving a 0.994 IoU on the test set.
* Built an algorithm that corrects the orientation of the image.
* Built a web app using Streamlit so that anyone can use the model to remove the background from their selfies.

## Main technologies used:

* Tensorflow
* Keras
* Numpy
* Matplotlib

## Main techniques used:

* Deep Learning
* Modern Computer Vision
* Data Augmentation
* Image Segmentation

## Visualizing the data:

I used the AISegment.com Matting Human dataset, which consists of approximately 34 thousand pictures and masks of mid-upper body selfies, with only one person in each picture, relatively near from the camera and with a high contrast with the background.

Let's see one image along with its provided label:

![descarga (6)](https://user-images.githubusercontent.com/70718425/107150396-e7fe2680-695d-11eb-904c-843b520366cf.png)

## Model Selection:

Original U-net architecture:

![descarga (7)](https://user-images.githubusercontent.com/70718425/107150849-2694e080-6960-11eb-810e-f66ef8dc9588.png)

I built and tested three U-net-like architectures, **achieving an IoU of 0.994 on the test set** with the last one.

## [Correcting the orientation:](https://github.com/javiergarciamolina/selfie-background-removal/blob/main/correct_image_orientation.ipynb)

After deploying the web app, I saw that sometimes the server would randomly rotate the image, which lead to bad results.

**So I built a deep learning model that would tell whether an image is correctly oriented, and an algorithm that would rotate it to its correct orientation.**

## Visualizing some results:

Let's see some examples of test images plotted against their ground-truth (or at least the one provided in the dataset) and their predictions:

![descarga (1)](https://user-images.githubusercontent.com/70718425/105999038-857b6f80-60ad-11eb-9bb1-f5fdf189d9bc.png)

## How could we improve this algorithm?

* **Getting different kinds of pictures (not only mid-upper body selfies).**
* Getting more and better labelled data.
* After getting more data, training more complex models and using ensemble-learning.


## Sources:

I used the article that describes the U-net architecture, it can be found here: https://arxiv.org/abs/1505.04597



