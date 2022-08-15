# venture_funding_with_deep_learning

## Prepare the Data for Use on a Neural Network Model
Using your knowledge of Pandas and StandardScaler from scikit-learn, preprocess the dataset so that you can later use it to compile and evaluate the neural network model. To do so, complete the following steps:

*   Read the applicants_data.csv file into a Pandas DataFrame. Review the DataFrame, checking for categorical variables that will need to be encoded and for columns that might eventually define your features and target variables.

*   Drop the “EIN” (Employer Identification Number) and “NAME” columns from the DataFrame, because they’re irrelevant for the binary classification model.

*   Encode the categorical variables of the dataset by using OneHotEncoder, and then place the encoded variables in a new DataFrame.

*   Add the numerical variables of the original DataFrame to the DataFrame that contains the encoded variables

*   Using the preprocessed data, create the features (X) and target (y) datasets. The “IS_SUCCESSFUL” column in the preprocessed DataFrame should define the target dataset. The remaining columns should define the features dataset.

*   Split the features and target datasets into training and testing datasets.

*   Use StandardScaler from scikit-learn to scale the features data.


## Compile and Evaluate a Binary Classification Model Using a Neural Network
Use your knowledge of TensorFlow to design a binary classification deep neural network model. This model should use the features of the dataset to predict whether a startup that’s funded by Alphabet Soup will become successful. Consider the number of inputs before determining both the number of layers that your model will contain and the number of neurons on each layer. Then compile and fit your model. Finally, evaluate your binary classification model to calculate the model’s loss and accuracy.

To do so, complete the following steps:

*   Use Tensorflow’s Keras to create a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer.

*   Compile and fit the model by using the binary_crossentropy loss function, the adam optimiser, and the accuracy evaluation metric.

*   Evaluate the model by using the test data to determine the model’s loss and accuracy.

*   Save and export your model to an HDF5 file, and name the file AlphabetSoup.h5.

## Optimise the Neural Network Model
Using your knowledge of TensorFlow and Keras, optimise your model to improve its accuracy. Even if you don’t achieve a better accuracy, you'll need to demonstrate at least two attempts to optimise the model. You can include these attempts in your existing notebook. Or, you can make copies of the starter notebook in the same folder, rename them, and code each model optimisation in a new notebook.

To do so, complete the following steps:

*   Define at least three new deep neural network models (that is, the original plus two optimisation attempts). For each, try to improve your first model’s predictive accuracy.

*   Adjust the input data by dropping different feature columns to ensure that no variables or outliers confuse the model.

*   Add more neurons (nodes) to a hidden layer.

*   Add more hidden layers.

*   Use different activation functions for the hidden layers.

*   Increase or reduce the number of epochs in the training regimen.

*   Display the accuracy scores that each model achieved, and then compare the results.

*   Save each model as an HDF5 file.

## Results

* Original Model:
  * Loss: 0.5543655753135681, Accuracy: 0.7303789854049683

* Alternate Model 1 - Added more nodes to hidden layer
  * Loss: 0.5563762187957764, Accuracy: 0.7300291657447815

* Alternate Model 2 - Reduced Epochs to 40
  * Loss: 0.552978515625, Accuracy: 0.72967928647995

* Alternate Model 3 - Added one more hidden layer
  * Loss: 0.5564121007919312, Accuracy: 0.7303789854049683
