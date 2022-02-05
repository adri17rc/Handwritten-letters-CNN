# Handwritten-letters-CNN
Recognition of handwritten letters using a Convolutional Neural Network

The following model trained a set of 600000 handwritten letters (latin alphabet) to recognise them and translate them into a digital file. The data was downloaded 
from a previously labeled set, called NIST. It can be dowloaded in the following link (https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format/metadata). The dataset came in a csv format, structured in 785 columns, where 784 correspond to the 28x28 images and the remaining one are the labels, 
that is, the letters. They are ordered alphabetically, so before going any step further the data was shuffled. Hadn't it been done, the model will not train the last latters in the alphabet, and would not work properly. The following images show an example of the letters used for the training, as well as their frequency. A quick look displays the difference of samples available on each letter; for example, the are almost 60000 'o' but just a few hundred 'i', This can affect the accuracy of the model, shwing a pauper performance in predictions of the most 'uncommon' ones. Further examples of these letters should be provided.

![Letters_NIST](https://user-images.githubusercontent.com/96789733/152639060-a4efebb3-4f5e-4a37-8145-50f078dcfc11.png)![Frequency_NIST](https://user-images.githubusercontent.com/96789733/152639253-d74ff671-3563-4c26-a88c-b91d4c66c69b.png)
