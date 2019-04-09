# TPU-UNET
TPU U-Net Cup Disc Segmentation
## TPU based UNET Segmentation.

The colaboratory demostrates  the segmentation of Fundus images using the very well known U-Net network in Keras.
This work is originally based on:
[Sevastopolsky A., Optic disc and cup segmentation methods for glaucoma detection with modification of U-Net convolutional neural network, Pattern Recognition and Image Analysis 27 (2017), no. 3, 618–624](https://github.com/seva100/optic-nerve-cnn )

The code was rewriten and adapted to TPU based trainning by Javier Civit and Anton Civit. The main modifications are:
* We use a completely different dual image generator and use it for both training and testing. For TPU training we need much larger static datasets and, thus we also make use of static data augmentation including images with modified brightness and modified parameters for the adaptive histogram equalization. This, together with the use of images from three different publicly available datasets for training and validation improves the robustness to the use if images acquired with different instruments.
* We use the version of Keras included in tensorflow. This is necessary to be able to execute in TPUs.
* We use [Pröve's parameterizable recursive U-net model](https://github.com/pietz/unet-keras). This model allows us to easily change many parameters necessary to compare different implementations of U-Net. Specifically, we can change the network depth and with, the use of drop out and batch normalization, the use of upsampling (although this type of layer is not currently supported by Keras in TPUs) nor transpose convolution and the width ratio between successive layers
* We use 120 image batches for both training and testing and train for 15 epochs using 150 training steps and 30 testing steps per epoch. We use an Adam optimizer algorithm in most cases with a .00075 learning rate although in a few cases, we have had to lower this value to ensure convergence. This values have proven suitable for TPU based training in U-Net architecture and provide good results with training times bellow 20 minutes even in the most complex implementations.
* Training is completely rewritten to run on TPUs
* The radii ratio parameter is calculated.

