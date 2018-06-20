import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

segment_size = 128
num_input_channels = 3
l2_reg = 5e-4
dropout_rate = 0.15
n_classes = 6

"""# Data Parameters
tf.flags.DEFINE_string("data test file", "C:/Python35/tmp/DataTest.npy", "Data source for testing")
tf.flags.DEFINE_string("labels_test_file", "C:/Python35/tmp/truth.npy", "Data source for the labels")

# Eval Parameters
tf.flags.DEFINE_integer("segment_size", 128, "Batch Size (default: 128)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")"""

def plot_cm(cM, labels,title):
    cmNormalized = np.around((cM/cM.sum(axis=1)[:,None])*100,2)
    fig = plt.figure()
    plt.imshow(cmNormalized,interpolation=None,cmap = plt.cm.Blues)
    plt.colorbar()
    plt.clim(0,100)
    plt.xlabel('Predicted Values')
    plt.ylabel('Ground Truth')
    plt.title(title + '\n%age confidence')
    plt.xticks(range(len(labels)),labels,rotation = 60)
    plt.yticks(range(len(labels)),labels)
    width, height = cM.shape
    print('Accuracy for each class is given below.')
    for predicted in range(width):
        for real in range(height):
            color = 'black'
            if(predicted == real):
                color = 'white'
                print(labels[predicted].ljust(12)+ ':', cmNormalized[predicted,real], '%')
            plt.gca().annotate(
                    '{:d}'.format(int(cmNormalized[predicted,real])),xy=(real, predicted),
                    horizontalalignment = 'center',verticalalignment = 'center',color = color)

    plt.tight_layout()
    fig.savefig(title +'.png')

class ImportGraph():
    """  Importing and running isolated TF graph """
    def __init__(self, loc):
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            saver = tf.train.import_meta_graph('model_cnn-15000' + '.meta',clear_devices=True)
            saver.restore(self.sess, 'model_cnn-15000')
            self.activation = tf.get_collection('prediction')[0]
            self.x_ = tf.get_collection('x_')[0]
            self.keep_prob = tf.get_collection('keep_prob')[0]
            self.prediction = self.graph.get_operation_by_name('prediction_opt').outputs[0]
            self.x_ = self.graph.get_operation_by_name('input_x').outputs[0]
            self.keep_prob = self.graph.get_operation_by_name('keep_prob_opt').outputs[0]

    def run(self, data):
        """ Running the activation operation previously imported """
        return self.sess.run(self.prediction , feed_dict={"input_x:0": data, "keep_prob_opt:0": 1.0})


labels = ['Standing','Standing_up','Sitting','Downstairs','Upstairs','Walking']

print('Importing Architecture, Parameters...')
data = np.load('DataTest.npy')
print(data.shape)
data = data.reshape([data.shape[0],384])
print('Number of samples tested :', data.shape[0])
print('segment length :', segment_size)
print('Reconstructing the architecture...')
model = ImportGraph('model_cnn-15000')
print('Working...')
predictions = model.run(data)
print('Output...')
print("")

groundTruth = np.load('truth.npy')
print(groundTruth.shape)
predictions = np.asarray(pd.get_dummies(predictions),dtype = np.int8)

groundTruth = np.argmax(groundTruth,1)
predictions = np.argmax(predictions,1)

print('Encoded Classes :')
print("")
print('- 0 : Standing')
print('- 1 : Standing_up')
print('- 2 : Sitting')
print('- 3 : Downstairs')
print('- 4 : Upstairs')
print('- 5 : Walking')
print("")
print("========================================")


True_value = {}.fromkeys(set(groundTruth),0)
for valeur in groundTruth :
    True_value[valeur]+=1

print("The groundtruth labels are given below :")
print(True_value)
print("")

compte = {}.fromkeys(set(predictions),0)
for valeur in predictions:
    compte[valeur]+=1

print("The predicted labels are given below:")
print(compte)
print("========================================")

cm = metrics.confusion_matrix(groundTruth,predictions)

print("")
plot_cm(cm, labels,'Confusion Matrix')
print("")
print(' --- Confusion matrix stored at the same path.---')
