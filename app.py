import tensorflow as tf
import gradio as gr
from urllib.request import urlretrieve

#urlretrieve("https://gr-models.s3-us-west-2.amazonaws.com/mnist-model.h5","mnist-model.h5")
model = tf.keras.models.load_model("mnist-model.h5")

def recognize_digit(inp):
    prediction = model.predict(inp.reshape(1, 28, 28, 1)).tolist()[0]
    return {str(i): prediction[i] for i in range(10)}

sketchpad = gr.inputs.Sketchpad()
label = gr.outputs.Label(num_top_classes=3)

gr.Interface(fn=recognize_digit, inputs=sketchpad,
  outputs=label, live=True).launch()