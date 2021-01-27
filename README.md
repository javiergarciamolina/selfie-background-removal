# Selfie-background-removal

## App:

You can try it at:

https://share.streamlit.io/javiergarciamolina/selfie-background-removal/main/background-removal-app.py

Please take into account that the model was trained on mid-upper body selfies, with only one person in the picture, relatively near from the camera and with a high contrast with the background.

## Objective:

In this project I wanted to build an algorithm that could automatically remove the background from a selfie. For that, I used the U-net architecture and the AISegment.com Matting Human dataset, which consists of approximately 34 thousand pictures of mid-upper body selfies, with only one person in each picture, relatively near from the camera and with a high contrast with the background.

## Result:

**The model achieved an Intersection over Union of 99.4 on the test set.**

Let's see some examples of test images plotted against their ground-truth (or at least the one provided in the dataset) and their predictions:

![descarga (1)](https://user-images.githubusercontent.com/70718425/105999038-857b6f80-60ad-11eb-9bb1-f5fdf189d9bc.png)

**How could we improve this algorithm?**


* Getting more and better labelled data.
* Getting different kinds of photos (not only mid-upper body selfies).
* After getting more data, training more complex models and using ensemble-learning.

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

## Sources:

I used article which describes the U-net architecture, it can be found here: https://arxiv.org/abs/1505.04597

## Model:

The final model was a U-net-like architecture, with less channels, batch normalization and dropout:

![descarga (13)](https://user-images.githubusercontent.com/70718425/105999962-95e01a00-60ae-11eb-982d-5535befd4d9b.png)


