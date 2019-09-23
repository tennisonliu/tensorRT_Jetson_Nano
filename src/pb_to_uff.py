import tensorflow as tf
from tensorflow.python.platform import gfile
import uff

def process_frozen_graph(pb_file_path):
	with tf.Session() as sess:
		print('loading graph')
		with gfile.FastGFile(pb_file_path, 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			sess.graph.as_default()
			tf.import_graph_def(graph_def, name='')
			writer = tf.summary.FileWriter("./graph_summary", graph=sess.graph)

			tensors = [op for op in sess.graph.get_operations()]    
			placeholders = [op for op in sess.graph.get_operations() if op.type == "Placeholder"]

			input_names = [op.name for op in placeholders]
                        
			input_shapes = [tf.get_default_graph().get_tensor_by_name(op.name+':0').get_shape() for op in placeholders]
			output_names = [tensors[-1].name]
			output_shapes = [tf.get_default_graph().get_tensor_by_name(tensors[-1].name+':0').get_shape().as_list()]

			print('input name: {}'.format(input_names))
			print('input shape: {}'.format(input_shapes))
			print('output name: {}'.format(output_names))
			print('output shape: {}'.format(output_shapes))
		
	return input_names, input_shapes, output_names, output_shapes

def convert_to_uff(frozen_graph, output_name, output_file_name='uff_model.uff'):
	uff_graph, graph_inputs, graph_outputs = uff.from_tensorflow_frozen_model(frozen_graph, output_name, output_filename = output_file_name, return_graph_info = True)
	return uff_graph

if __name__ =='__main__':
	model_path = './model/mobileNet.pb'
	input_names, input_shapes, output_names, output_shapes = process_frozen_graph(model_path)
	# uff_graph = convert_to_uff(model_path, output_names)
