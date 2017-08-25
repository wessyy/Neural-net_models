import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import csv

'''
input > weight > HL1 (activation function) > weights > HL2 (activation function) > weights > output layer

compare output to intended output > cost or loss function (cross entropy)
Optimization function > minimize cost (AdamOptimizer...Stochastic gradient descent): Backpropagation

feed forward + backprop = epoch
'''

''' Reading Dataset'''
trained_model = "ac_line_gen_14/new.ckpt"
n_classes = 5
dataset = pd.read_csv('ac_line_gen_14/ac_line_gen_14.csv', header=0, sep=',')
X, y = dataset.iloc[:,:-n_classes].astype(np.float).values, dataset.iloc[:, -n_classes:].astype(np.float).values


# Formatting and Normalization
y = np.reshape(y, [y.shape[0], n_classes])
scaled_X = preprocessing.scale(X)

# Splitting training and testing
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=1)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25, random_state=0)


''' Parameters '''
save_model = True
display_step = 10
learning_rate = 0.003
hm_epochs = 10
batch_size = 100
dropout_rate = 1.0 # Probability to keep units
num_features = X_train.shape[1]
total_len = X_train.shape[0]

layers = [num_features, 50, 400, 500, 500, n_classes]
#500, 600, 600, 600, 600


x = tf.placeholder('float', [None, num_features])
y = tf.placeholder('float', [None, n_classes])
keep_prob = tf.placeholder('float') # dropout (keep probability)

#Create weight and bias vectors for an MLP
#layers: The number of neurons in each layer (including input and output)
def createWeights(layers):
    weights = []
    for i in range(len(layers) - 1):
        #Fan-in for layer; used as standard dev
        # lyrstd = np.sqrt(1.0 / layers[i])
        curW = tf.Variable(tf.random_normal([layers[i], layers[i + 1]], 0, 0.1))
        weights.append(curW)
    return weights

def createBiases(layers):
    biases = []
    for i in range(len(layers) - 1):
        #Fan-in for layer; used as standard dev
        # lyrstd = np.sqrt(1.0 / layers[i])
        curB = tf.Variable(tf.random_normal([layers[i + 1]], 0, 0.1))
        biases.append(curB)
    return biases

def multilayer_perceptron(data, dropout, weights, biases):
    for i in range(len(weights)-1):
        data = tf.nn.relu(tf.add(tf.matmul(data, weights[i]), biases[i]))
        
    data = tf.nn.dropout(data, dropout)
    return tf.matmul(data, weights[len(weights)-1]) + biases[len(weights)-1]

weights = createWeights(layers)
biases = createBiases(layers)

saver = tf.train.Saver()

def train_neural_network(x):
    # Initialize Variables
    prediction = multilayer_perceptron(x, keep_prob, weights, biases)
    cost = tf.reduce_mean(tf.square(prediction - y))
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        epoch = 1

        best_train_error = best_validation_error = 10000000
        total_iterations = 0

        while epoch <= hm_epochs:
            avg_epoch_loss = 0
            total_batch = int(total_len/batch_size)

            for i in range(total_batch-1):
                epoch_x = X_train[i*batch_size:(i+1)*batch_size]
                epoch_y = y_train[i*batch_size:(i+1)*batch_size]

                # Run optimization (backprop) and cost op (to get loss)
                _, c = sess.run([optimizer, cost], feed_dict = {x:epoch_x, y:epoch_y, keep_prob: dropout_rate})
                error = sess.run(cost, feed_dict = {x:epoch_x, y:epoch_y, keep_prob: 1.0})
                p = sess.run(prediction, feed_dict={x: epoch_x, keep_prob: 1.0})

                avg_epoch_loss += error / total_batch

            validation_loss = sess.run(tf.cast(cost, 'float'), feed_dict={x:X_validation, y:y_validation, keep_prob:1.0})
            validation_loss = tf.convert_to_tensor(validation_loss).eval({x:X_validation, y:y_validation, keep_prob:1.0})

            # Display predictions
            print('Epoch', epoch, 'completed out of', hm_epochs, 'training loss:', avg_epoch_loss, 'validation loss:', validation_loss)
            if epoch % display_step == 0:
                for i in range(3):
                    print ("Label value:", epoch_y[i], "Estimated value:", p[i])
                print ("[*]============================")

            if save_model:
                if avg_epoch_loss < best_train_error:
                    # print("New best error -- Training Set MSE: ", avg_epoch_loss, "| Validation Set MSE: ", validation_loss)
                    best_train_error = avg_epoch_loss
                    saver.save(sess, trained_model)


            epoch += 1

        print('Best Mean Squared Error: ', best_train_error)
        test_neural_network(X_test, y_test, trained_model)
  


''' Find testing accuracy '''
def test_neural_network(features, labels, model_file):
    prediction = multilayer_perceptron(x, keep_prob, weights, biases)
    cost = tf.reduce_mean(tf.square(prediction - y))


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_file)

        accuracy = sess.run(tf.cast(cost, 'float'), feed_dict={x:features, y:labels, keep_prob:1.0})
        mse = tf.convert_to_tensor(accuracy).eval({x:features, y:labels, keep_prob:1.0})
        print('Mean Squared Error on Test Set:', mse)
        
        return mse


''' Make chart of errors '''
def output_errors(features, labels, model_file):
    prediction = multilayer_perceptron(x, keep_prob, weights, biases)
    cost = tf.reduce_mean(tf.sqrt(tf.square(prediction - y)))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_file)
        # mse_lst = []
        
        with open('ac_gen_118/ac_gen_118_actvspred1.csv', 'wb') as csv_file:
            writer = csv.writer(csv_file)
            for i in range(len(features)):
                # accuracy = sess.run(tf.cast(cost, 'float'), feed_dict={x:[features[i]], y:[labels[i]], keep_prob:1.0})
                # mse = tf.convert_to_tensor(accuracy).eval({x:[features[i]], y:[labels[i]], keep_prob:1.0})
                # mse_lst.append(mse)
                p = sess.run(tf.cast(prediction, 'float'), feed_dict={x:[features[i]], keep_prob:1.0})
                writer.writerow([p[0][10], labels[i][10]])

''' Make predictions with MLP model '''
# features should be [[feature1, feature2, feature3, ...]]
def use_neural_network(features, model_file):
    prediction = multilayer_perceptron(x, 1.0, weights, biases)
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_file)

        features_scaled = preprocessing.scale(np.array(list(features)))
        result = (sess.run(prediction, feed_dict={x:features_scaled}))

        print(result)
        return result


if __name__ == "__main__":
    # train_neural_network(x) 
    test_neural_network(X_test, y_test, trained_model)
    # output_errors(X_test, y_test, trained_model)
    # use_neural_network([[11.528,74.347,59.098,9.5418,9.378,16.085,12.246,4.6638,7.9328,19.916,20.664,14.864,27.466,-3.9764,0.96155,3.8931,22.382,6.3405,2.3666,1.3805,7.7978,4.6675,0.053795,0.29828,0.012721,0.010129,0.01012]], trained_model)
    


















