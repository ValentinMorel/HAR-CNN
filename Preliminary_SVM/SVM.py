import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
ops.reset_default_graph()

sess = tf.Session()

data = pd.read_csv("C:/Python35/acquis_smartphone/dataset_class_3axes_118_118.txt", sep = ";")
train = data.values[1:-1, 0:11]
target = data.values[1:-1, 12]
target = target.astype(np.float32)
train = train.astype(np.float32)

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
train=sel.fit_transform(train)


x_vals = np.array([[x[0], x[2]] for x in train])
y_vals1 = np.array([1 if y==0 else -1 for y in target])
y_vals2 = np.array([1 if y==1 else -1 for y in target])
y_vals3 = np.array([1 if y==2 else -1 for y in target])
y_vals4 = np.array([1 if y==3 else -1 for y in target])
y_vals = np.array([y_vals1, y_vals2, y_vals3, y_vals4])
class1_x = [x[0] for i,x in enumerate(x_vals) if target[i]==0]
class1_y = [x[1] for i,x in enumerate(x_vals) if target[i]==0]
class2_x = [x[0] for i,x in enumerate(x_vals) if target[i]==1]
class2_y = [x[1] for i,x in enumerate(x_vals) if target[i]==1]
class3_x = [x[0] for i,x in enumerate(x_vals) if target[i]==2]
class3_y = [x[1] for i,x in enumerate(x_vals) if target[i]==2]
class4_x = [x[0] for i,x in enumerate(x_vals) if target[i]==3]
class4_y = [x[1] for i,x in enumerate(x_vals) if target[i]==3]


batch_size = 50

x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[4, None], dtype=tf.float32)
prediction_grid = tf.placeholder(shape=[None, 2], dtype=tf.float32)

# Create variables for svm
b = tf.Variable(tf.random_normal(shape=[4,batch_size]))

# Gaussian (RBF) kernel
gamma = tf.constant(-5.0)
dist = tf.reduce_sum(tf.square(x_data), 1)
dist = tf.reshape(dist, [-1,1])
sq_dists = tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))
my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

# Declare function to do reshape/batch multiplication
def reshape_matmul(mat):
    v1 = tf.expand_dims(mat, 1)
    v2 = tf.reshape(v1, [4, batch_size, 1])
    return(tf.matmul(v2, v1))

# Compute SVM Model
first_term = tf.reduce_sum(b)
b_vec_cross = tf.matmul(tf.transpose(b), b)
y_target_cross = reshape_matmul(y_target)

second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)),[1,2])
loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))

# Gaussian (RBF) prediction kernel
rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1),[-1,1])
rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1),[-1,1])
pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))), tf.transpose(rB))
pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

prediction_output = tf.matmul(tf.multiply(y_target,b), pred_kernel)
prediction = tf.arg_max(prediction_output-tf.expand_dims(tf.reduce_mean(prediction_output,1), 1), 0)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target,0)), tf.float32))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
batch_accuracy = []
for i in range(5000):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = x_vals[rand_index]
    rand_y = y_vals[:,rand_index]
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    
    acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x,
                                             y_target: rand_y,
                                             prediction_grid:rand_x})
    batch_accuracy.append(acc_temp)
    
    if (i+1)%100==0:
        print('Step #' + str(i+1))
        print('Loss = ' + str(temp_loss))

# Create a mesh to plot points in
x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_predictions = sess.run(prediction, feed_dict={x_data: rand_x,
                                                   y_target: rand_y,
                                                   prediction_grid: grid_points})
grid_predictions = grid_predictions.reshape(xx.shape)

# Plot points and grid
plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8)
plt.plot(class1_x, class1_y, 'ro', label='Standing')
plt.plot(class2_x, class2_y, 'kx', label='Walking')
plt.plot(class3_x, class3_y, 'gv', label='Downstairs')
plt.plot(class4_x, class4_y, 'b.', label='Upstairs')
plt.title('Gaussian SVM Results on Human Activity Dataset ')
plt.xlabel('Features Values')
plt.ylabel('Activity')
plt.legend(loc='lower right')
plt.ylim([-1, 7.0])
plt.xlim([-1, 7])
plt.show()

# Plot batch accuracy
plt.plot(batch_accuracy, 'k-', label='Accuracy')
plt.title('Batch Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

print(max(batch_accuracy*100))

