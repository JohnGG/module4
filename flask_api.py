from flask import Flask
import json
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# This could be added to the Flask configuration
MODEL_PATH = './model.pb'

# Read the graph definition file
with open(MODEL_PATH, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# Load the graph stored in `graph_def` into `graph`
graph = tf.Graph()
with graph.as_default():
    tf.import_graph_def(graph_def, name='')
    nodes = [n.name for n in tf.get_default_graph().as_graph_def().node]
    for no in nodes:
        print(no)

# Enforce that no new nodes are added
graph.finalize()

# Create tensorflow session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(graph=graph, config=config)


# Get the input and output operations
input_op = graph.get_operation_by_name('inputs')
input_tensor = input_op.outputs[0]
output_op = graph.get_operation_by_name('pred')
output_tensor = output_op.outputs[0]


@app.route("/predict")
def company():
    try:
        # Generate placeholders nodes
        placeholders = {input_tensor: [np.zeros(shape=(50, 50, 3))]}
        preds = sess.run(output_tensor, placeholders)
        print(preds)
        return json.dumps({})
    except KeyError:
        return json.dumps({'error': 'Company not found', 'code': 'err_not_found'}), 404


if __name__ == "__main__":
    app.run(debug=True)