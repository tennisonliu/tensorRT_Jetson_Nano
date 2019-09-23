import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import sys
import uff
import argparse
sys.path.insert(0, 'src/')
import tensorNet
import cv2
from tensorflow.keras.applications.mobilenet import MobileNet as Net
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions

OUTPUT_LEN = 1001

parser = argparse.ArgumentParser()
parser.add_argument("img_path", help="path to input image",
					type=str)
args = parser.parse_args()
print(args.img_path)
'''
print('Creating UFF File from MobileNet')
sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) # CPU only
K.set_session(sess)
K.set_learning_phase(0)
model = Net(weights='imagenet')
K.set_learning_phase(0)
output_name = model.output.op.name
input_name = model.input.op.name
frozen_graph = tf.graph_util.remove_training_nodes(tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output_name]))
'''

# Convert Tensorflow graph to UFF file
uff_model = uff.from_tensorflow('./model/mobileNet.pb', ['MobilenetV2/Predictions/Reshape_1'], output_filename='./model/uff_model.uff')


engine = tensorNet.createTrtFromUFF('./model/uff_model.uff')
tensorNet.prepareBuffer(engine)
print('Successfully created TensorRT engine.')


print('Performing inference on input {}: ...'.format(args.img_path))
img_height = 224
img = image.load_img(args.img_path, target_size=(img_height, img_height))
# converts into 3D Numpy Array
_input = image.img_to_array(img)
_input = np.expand_dims(_input, axis=0)
_input = preprocess_input(_input)

print(_input)
print(_input.shape)


_output = np.zeros([OUTPUT_LEN], np.int32)
tensorNet.inference(engine, _input, _output)
print(_output.tolist())