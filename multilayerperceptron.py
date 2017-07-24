import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

'''
input > weight > HL1 (activation function) > weights > HL2 (activation function) > weights > output layer

compare output to intended output > cost or loss function (cross entropy)
Optimization function > minimize cost (AdamOptimizer...Stochastic gradient descent): Backpropagation

feed forward + backprop = epoch
'''

''' Reading Dataset'''
dataset = pd.read_csv('max_price_diff/max_price_diff.csv', header=None, sep=',')[1:]
X, y = dataset.iloc[:,:-1].astype(np.float).values, dataset.iloc[:, -1].astype(np.float).values

# Formatting and Normalization
y = np.reshape(y, [y.shape[0], 1])
scaled_X = preprocessing.scale(X)

# Splitting training and testing
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=2)
total_len = X_train.shape[0]

''' Parameters '''
display_step = 1
learning_rate = 0.003
hm_epochs = 500
batch_size = 100
dropout_rate = 1.0 # Probability to keep units
n_nodes_hl1 = 50
n_nodes_hl2 = 400
n_nodes_hl3 = 500
n_nodes_hl4 = 500
n_classes = 1
num_features = X_train.shape[1]


x = tf.placeholder('float', [None, num_features])
y = tf.placeholder('float', [None, 1])
keep_prob = tf.placeholder('float') # dropout (keep probability)

weights = {
    'h1': tf.Variable(tf.random_normal([num_features, n_nodes_hl1], 0, 0.1)),
    'h2': tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2], 0, 0.1)),
    'h3': tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3], 0, 0.1)),
    'h4': tf.Variable(tf.random_normal([n_nodes_hl3,n_nodes_hl4], 0, 0.1)),
    'out': tf.Variable(tf.random_normal([n_nodes_hl4, n_classes], 0, 0.1))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_nodes_hl1], 0, 0.1)),
    'b2': tf.Variable(tf.random_normal([n_nodes_hl2], 0, 0.1)),
    'b3': tf.Variable(tf.random_normal([n_nodes_hl3], 0, 0.1)),
    'b4': tf.Variable(tf.random_normal([n_nodes_hl4], 0, 0.1)),
    'out': tf.Variable(tf.random_normal([n_classes], 0, 0.1))
}

def multilayer_perceptron(data, dropout):
	l1 = tf.add(tf.matmul(data, weights['h1']), biases['b1'])
	l1 = tf.nn.relu(l1)
	l1 = tf.nn.dropout(l1, dropout)

	l2 = tf.add(tf.matmul(l1, weights['h2']), biases['b2'])	
	l2 = tf.nn.relu(l2) 
	l2 = tf.nn.dropout(l2, dropout)

	l3 = tf.add(tf.matmul(l2, weights['h3']), biases['b3'])
	l3 = tf.nn.relu(l3) 
	l3 = tf.nn.dropout(l3, dropout)

	l4 = tf.add(tf.matmul(l3, weights['h4']), biases['b4'])
	l4 = tf.nn.relu(l4) 
	l4 = tf.nn.dropout(l4, dropout)

	output = tf.matmul(l4, weights['out']) + biases['out']
	return output

saver = tf.train.Saver()
tf_log = 'tf3.log'

def train_neural_network(x):
	prediction = multilayer_perceptron(x, keep_prob)
	cost = tf.reduce_mean(tf.square(prediction - y))

	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# try:
		# 	epoch = int(open(tf_log,'r').read().split('\n')[-2]) +1
		# 	print('STARTING:',epoch)
		# except:
		epoch = 1

		best_loss = 10000000
		total_iterations = 0

		while epoch <= hm_epochs:
			# if epoch != 1:
			# 	saver.restore(sess, "mlp_model.ckpt")
			avg_epoch_loss = 0
			total_batch = int(total_len/batch_size)

			for i in range(total_batch-1):
				epoch_x = X_train[i*batch_size:(i+1)*batch_size]
				epoch_y = y_train[i*batch_size:(i+1)*batch_size]

				# Run optimization (backprop) and cost op (to get loss)
				_ , c = sess.run([optimizer, cost], feed_dict = {x:epoch_x, y:epoch_y, keep_prob: dropout_rate})
				p = sess.run(prediction, feed_dict={x: epoch_x, keep_prob: 1.0})

				avg_epoch_loss += c / total_batch

			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', avg_epoch_loss)
			
			label_val = epoch_y
			estimate = p

			if epoch % display_step == 0:
				for i in range(3):
					print ("label value:", label_val[i], "estimated value:", estimate[i])
				print ("[*]============================")

			if avg_epoch_loss < best_loss:
				best_loss = avg_epoch_loss
				saver.save(sess, "mlp_model_gen_diff.ckpt")
				with open(tf_log,'a') as f:
					f.write(str(epoch)+'\n') 

			epoch += 1

		print('Best Mean Squared Error: ', best_loss)

		accuracy = sess.run(tf.cast(cost, 'float'), feed_dict={x:X_test, y:y_test, keep_prob:1.0})
		mse = tf.convert_to_tensor(accuracy).eval({x:X_test, y:y_test, keep_prob:1.0})
		print('Mean Squared Error on Last Set:', mse)

		test_neural_network(X_test, y_test, "mlp_model_gen_diff.ckpt")
		

''' Find testing accuracy '''
def test_neural_network(features, labels, model_file):
	prediction = multilayer_perceptron(x, keep_prob)
	cost = tf.reduce_mean(tf.square(prediction - y))

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, model_file)

		accuracy = sess.run(tf.cast(cost, 'float'), feed_dict={x:features, y:labels, keep_prob:1.0})
		mse = tf.convert_to_tensor(accuracy).eval({x:features, y:labels, keep_prob:1.0})
		print('Mean Squared Error on Test Set:', mse)

		# return mse


''' Make predictions with MLP model '''
def use_neural_network(features, model_file):
    prediction = multilayer_perceptron(x, keep_prob)
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_file)

        features_scaled = preprocessing.scale(np.array(list(features)))
        result = (sess.run(prediction, feed_dict={x:features_scaled}))

        print(result)


if __name__ == "__main__":
	# train_neural_network(x)	

	test_neural_network(X_test, y_test, "max_price_diff/mlp_model_price.ckpt")
	

















