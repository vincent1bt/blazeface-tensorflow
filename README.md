# Face Detection From Scratch using the BlazeFace Architecture

In this repository you can find the code to test and train a face detection model based on the BlazeFace architecture.

This model can run on even low-end hardware like a 2 core CPU. You can learn more about the architecture and the process to train the network in [this post](https://vincentblog.link/posts/face-detection-for-low-end-hardware-using-the-blaze-face-architecture).

You can test the model in your computer using:

```
python webcam.py
```

The model is saved in the **weights** folder. You **need** TensorFlow 2.0, cv2, numpy and imutils installed.

## Create dataset

To train this model you need to create a dataset:

1. Firstly, download the **WIDER FACE** training dataset [here](http://shuoyang1213.me/WIDERFACE/) and the **FDDB** dataset [here](http://vis-www.cs.umass.edu/fddb/).

2. Create a folder named **face_dataset** and put inside the folder **originalPics** from the **FDDB** dataset and the folders **from WIDER_train/images** fromm the  **WIDER FACE** training dataset.

3. Create a folder named **created_dataset**

## Training

To train the network we run:

```
python training.py --epochs=500
```

500 epochs are enough to converge, if you want to keep training use:

```
python training.py --epochs=100 --continue_training
```

To test the trained model:

```
python test_model.py
```

This is only going to show the first 8 images, If you want to test more images you can use the Jupyter Notebook or create a new one and copy the code from the test_model.py file.

## iOS Apps

This model is also available as an iOS app written in Swift, actually, as 2 iOS apps. The first app uses **TensorFlow lite** to load and run the model, and the second app uses **CoreML**.

You can check the post about the **TensorFlow Lite** version [here](https://vincentblog.link/posts/blaze-face-in-i-os-using-tensor-flow-lite), and the **CoreML** version [here](https://vincentblog.link/posts/blaze-face-in-i-os-using-core-ml).

