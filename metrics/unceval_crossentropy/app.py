import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("unceval_crossentropy")
launch_gradio_widget(module)
