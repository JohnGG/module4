from flask import Flask, request
import json
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

dict_map = ["No boat", "Boat"]

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
        # Get image
        image_name = request.args["image_name"]
        image = cv2.imread(os.path.join("images", image_name))
        resized_image = cv2.resize(image, (50, 50))

        # Generate placeholders nodes
        placeholders = {input_tensor: [resized_image]}
        preds = sess.run(output_tensor, placeholders)

        return json.dumps({"prediction": dict_map[preds[0]]})
    except KeyError:
        return json.dumps({'error': 'Company not found', 'code': 'err_not_found'}), 404


if __name__ == "__main__":
    app.run(debug=True)