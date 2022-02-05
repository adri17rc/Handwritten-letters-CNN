# Handwritten-letters-CNN
Recognition of handwritten letters using a Convolutional Neural Network

The following model trained a set of 600000 handwritten letters (latin alphabet) to recognise them and translate them into a digital file. The data was downloaded 
from a previously labeled set, called NIST. It can be dowloaded in the following link (https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format/metadata). The dataset came in a csv format, structured in 785 columns, where 784 correspond to the 28x28 images and the remaining one are the labels, 
that is, the letters. They are ordered alphabetically, so before going any step further the data was shuffled. Hadn't it been done, the model will not train the last latters in the alphabet, and would not work properly. The following images show an example of the letters used for the training, as well as their frequency. A quick look displays the difference of samples available on each letter; for example, the are almost 60000 'o' but just a few hundred 'i', This can affect the accuracy of the model, shwing a pauper performance in predictions of the most 'uncommon' ones. Further examples of these letters should be provided.

![Letters_NIST](https://user-images.githubusercontent.com/96789733/152639060-a4efebb3-4f5e-4a37-8145-50f078dcfc11.png)![Frequency_NIST](https://user-images.githubusercontent.com/96789733/152639253-d74ff671-3563-4c26-a88c-b91d4c66c69b.png)

Data had to be prepared before training the model. The images were resized to 28x28 arrays in greyscale. Meanwhile, 'y' labels were map from the letters to their positions in the alphabet (0-26) and then changed to categorical values. 

-MODEL 
A sequential struture was chosen, consisting on 3 hidden layers, one Conv2D with kernel size 3x3, and 2 Dense layers of 90 and 26 nodes respectively. The last layer activation function is 'softmax', which returns the probability of each letter. Besides, a MaxPooling layer and Flatten were added after the convolutional layer, to optimize the training of the model. This resulted in 489624 trainable parameters. The optimizer chosen was 'adam' and the loss 'categorical_crossentropy'. It was fit through 8 epochs, with a batch_size of 200. For more details, consult the file NIST.py attached in the repository.

-RESULTS

After training the model, an accuracy of 0.9957 wa registered, and a loss of 0.0127. Thus, we can conclude that our model was trained succesfully and will predict, in most of the cases, the right letter. 

![ModelLoss_NIST](https://user-images.githubusercontent.com/96789733/152639941-a2766830-ac4c-41ea-b0d6-6a301928bc89.png)![ModelAccuracy_NIST](https://user-images.githubusercontent.com/96789733/152639943-44e6906a-67e0-4014-a2dc-1432d6bf2cec.png)

Finally, the first 100 hundred picutures were picked up and use to predict the letters. A 100% of acccuracy was obtained. 


