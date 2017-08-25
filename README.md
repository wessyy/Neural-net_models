# Neural-net_models

<p align="center"><b>Overview</b> </p>

The process of training and testing the neural network follows a few general steps:
1)	Collect the sample data.
2)	Format/Preprocess the sample data.
3)	Split the sample data into training, validation, and testing sets (done automatically by the program).
4)	Set any parameters needed.
5)	Run function to train the NN on the sample data.
6)	Test trained NN model.


<p align="center"><b>Collecting Data</b> </p>

The way I have collected the data so far is through the compare_ac_dc_solutions.m Matlab file. In the code of the file, you can:

    •	Set which bus case you want to run it on.
        o	This is done by setting the mpc variable to the bus case file name.
    •	The rate of perturbation desired, which category of data you want to extract (features and ouputs).
        o	This is done by finding the right indices in the different AC and DC matrices, and putting them in the data array to be written to a CSV.
        o	Note: The data should always follow the format of [Feature1, Feature2, …, Labels]. The features are what we are using as input and the labels are what we are trying to predict. For example, if we are trying to predict the power generation of each bus in the AC model (AC_gen), using the powers at each load as inputs (P_load), our data array should look like: [P_load1, P_load2, …, AC_gen1, AC_gen2, …]
    •	The location or file to write the data to. 
        o	This is done by setting the desired directory/file name in the second to last line of the code (in the dlmwrite function)
    •	How many iterations you’d like to run (how many samples you want to make). 
        o	This is done by setting the number of iterations in the top of the for loop.


<p align="center"><b>Format/Preprocess Data</b> </p>


Not much should need to be done here if you are using the Matlab file to generate the data. The big thing is to make sure the data follows the [Feature1, Feature2, …, Labels] format. Sometimes there are features that just all 0’s (if you open the csv in Excel, the entire column is 0’s). I will typically go and delete that entire column of features if they are all 0’s as they can just provide unnecessary noise.





<p align="center"><b>The Neural Network</b> </p>

This is the meat of this project. All the necessary code is located in the multilayerperceptron.py file. The NN code can train a model on a certain dataset, save the trained model, and call the trained model to make predictions. The code is split into a few different sections.

   -	**Requirements**

In order to run the script, some packages must be installed on your computer. The code uses Python 3.5, which can be installed here https://www.python.org/downloads/

Other than that, all other packages should be installed already on the virtual environment that is set up. So make sure that prior to running multilayerperceptron.py, you activate the virtual environment first by following these few steps:

1)	Open the command line.
2)	cd to the directory of the code.
3)	While in the directory, type the following into the command line: $ source venv/bin/activate
4)	You should now see (venv) before your command line prompt. This tells you the virtual environment was activated properly. Once here, everything else should run smoothly.

 -	**Reading Dataset**

At the top of the code there is a section commented “Reading Dataset.” This is where you input certain file names that are needed for the code to run. The code will automatically split the dataset into a training. testing, and validation set in a 60/20/20 split. 

Each set is assigned to the self-described variable: X_train, X_test, and X_validation are the inputs (or features) for the training, testing, and validation sets respectively, and the same for y_train, y_test, and y_validation, except those are the labels (or what we’re trying to predict. 

The following are the variables that need to be set:
    o	trained_model : This is the directory/file name of a model you want to save the NN you’re going to train to, or the model you want to use to make predictions or calculate testing errors. The file is always to .ckpt file. So you can name it anything as long as it has a .ckpt extension. Note that if the file name exists already, and you are training a new NN, the model will be overwritten to the new NN being trained.
    o	n_classes : This is the number of outputs that we are predicting.
    o	dataset : change the file name within the pd.read_csv function to the directory/file name of the dataset you will be training or testing on.


 -	**Setting Parameters**

These are certain parameters that determine how the code runs and how the neural network is built/trained. 

    o	save_model : Set to True if you want to save the NN model you are about to train to a .ckpt file, False otherwise.
    o	display_step : As the NN is training, it will output the MSE of each epoch it runs through. For each display_step, it will also output three sample’s labels and the predictions for those samples formed by the NN at that epoch. So for example,  if display_step was set to 10, then the labels vs predictions will be outputted every 10th epoch.
    o	learning_rate : The learning rate of the NN. Typically .003 tends to work the best, but you can play around with this to see what works best for the data.
    o	hm_epochs : How many epochs you want to train on.
    o	batch_size : Number of samples propagated through the network at one time.
    o	dropout_rate : Probability of keeping a unit. Used for dropout to avoid overfitting.
    o	num_features : Number of features you have in your data set. This will be automatically set as all the columns of data that are not a label/output.
    o	total_len : Ignore
    o	layers : This is an array of integers. The first element of the array is always num_features and the last element is always n_classes. The numbers in between represent the number of neurons in each hidden layer. To change the number of neurons, just simply change the number in the specified layer. To add or subtract the number of layers, simply add another element or take away an element from the array. For example, if I want 4 hidden layers with 50, 500, 500, and 500 neurons in each layer, my layers array would be [num_features, 50, 500, 500, 500, n_classes]. If I wanted 5 hidden layers with 50, 500, 500, 500, and 600 neurons, my layers array would be [num_features, 50, 500, 500, 500, 600, n_classes].
        ♣	Note: For the 14 and 30 bus model, [num_features, 50, 400, 500, 500, n_classes] seems to work really well. For the 118 bus model, I experimented with a lot of different size models, [num_features, 300, 500, 600, 600, 600, n_classes] seemed to work best.
    o	x, y, and keep_prob are simply Tensorflow placeholders and can be ignored. These are automatically set.


 -	**Training the Neural Network**

After all the parameters are set, the Neural Network should train properly. The function that trains the NN is train_neural_network(x). To call the function and to begin training, simply write train_neural_network(x) on the bottom of the script under 
if __name__ == “__main__”: 
	train_neural_network(x)

The NN should begin training immediately, and on each epoch will output a line like so:
“Epoch 1 completed out of 500 training loss: 100”
Every specified display_step it will output three lines of something like:
“Label value: [1, 2, 3]  Estimated value: [.999, 2.1, 3.01]”

When the NN has finished training, it will output the best MSE that it was able to acquire on the training set, and also output the MSE on the testing set. 

 -	**Testing the NN Model**

If you want to see the errors on different testing sets, simply call the test_neural_network(features, labels, model_file) function under if __name__ == “__main__”: 

What should be passed in as features is your data of features, labels are your data of labels, and model_file is the directory/file name of the saved .ckpt file of the trained NN model you want to use to test. For example, if I wanted to output the MSE of the testing set, I would call:

test_neural_network(X_test, y_test, trained_model)

Keep in mind that when testing your neural network, the layers array must be the same as what it was when you trained the neural network. Otherwise it will throw an error that dimensions don't match.

 -	**Making Predictions**

In order to see the actual prediction your NN model would yield, you can use the function use_neural_network(features, model_file).

This function takes in an array of features, and the directory/file name of the saved .ckpt file of the trained NN model, and will make calculate a prediction based on the features. This function only takes in one sample, so if you want to see predictions on multiple samples, you will have to call this in a loop. 














<p align="center"><b>Future Work</b> </p>


So far, we’re able to yield very low errors for small bus size systems like the 14 and 30 bus system. That is promising and shows a NN model can do well in predictions of this nature. The problem right now is that scaling up to larger size systems like the 118 bus system generates lot larger errors. 

I found that a number of generators in the 118 bus system generate no power most of the time, but occasionally spike up and generate power. These generators are extremely hard to predict for the NN, but the NN does a great job for generators that do generate power consistently. 

So what to do next:
-	How can we make the feature space constant, and not grow as much as the network grows?
-	How can we split the network model into sub-networks in order to make the problem smaller and easier to solve?
-	How can we determine the most important and predictive features, and exclude the ones that are not important?









