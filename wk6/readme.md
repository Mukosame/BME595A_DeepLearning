# Homework 6

_Neural Networks - CNN in PyTorch_

Xiaoyu Xiang(xiang43@purdue.edu)
--------------------
## Platform and Packages
- Python 3.6.5 
- PyTorch 0.4.0, torchvision
- Numpy, matplotlib, OpenCV 3

## Dataset: Tiny ImageNet

Tiny Imagenet has 200 classes. Each class has 500 training images, 50 validation images, and 50 test images. The training and validation sets with images and annotations. We provide both class labels and bounding boxes as annotations are providesd. In Tiny ImageNet challenge, we are asked only to predict the class label of each image without localizing the objects. The test set is released without labels. 

To load the dataset, we use todo

## Code Implementation

### train.py

It receives 2 args: --data and --save, that provides the directory path to for data and model.

```python
python3 train.py --data /tiny/imagenet/dir/ --save /dir/to/save/model/
```
The structure of AlexNet is shown below:

![AlexNet](https://github.com/Mukosame/BME595A_DeepLearning/blob/master/wk6/AlexNet.JPG "AlexNet")

To start with, we load the pretrained model in PyTorch and then train the last classifier layer based on it.

We set the following parameters for training this neural network:

- Epoch Number: 20
- Loss Function: MSE
- Start Learning Rate: 0.001, decrease by 10 at epoch 11, 15
- Batch Size: 128
- Optimizer: Adam
- Image Normalization: True
- Trained on: single GPU (Nvidia Quadro M6000)



## Test Result

Run ```python3 test.py --model /dir/containing/model/```, and we get the following results:

####Training and Validation Loss

The training and validation loss over epochs are shown below:

![My loss](https://github.com/Mukosame/BME595A_DeepLearning/blob/master/wk6/train_loss.jpg "training loss")

After training, we select the model with highest accuracy for visualization:

- At which epoch: 18
- Training Loss: 4.9497
- Training Accuracy: 35.7900%

When given an image(encoded in RGB and turned into numpy array), the predicted output of our model is shown as below:

![Test image's output](https://github.com/Mukosame/BME595A_DeepLearning/blob/master/wk6/view_result.jpg "Test image's output")

Since I don't have a webcamera, I cannot show the result related to it. Generally, the idea is to get the image from webcam and send it to ```net.view```. If the image is loaded correctly, then the result should be the same as above.

## Summary
- By loading the pretrained model as the initial value, we can ensure the training loss is in a reasonable range and thus greatly save the time of training
- When saving the weights, we should pay attention to the convert between cuda.Tensor and Tensor

## Reference
- Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems. 2012.
- https://tiny-imagenet.herokuapp.com/
- https://pytorch.org/docs/stable/torchvision/models.html
