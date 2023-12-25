import json

import cv2
import gradio as gr
import pandas as pd
import numpy as np
import os

from pathlib import Path

from tqdm import tqdm
from mltu.configs import BaseModelConfigs
from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer
from inferenceModel import ImageToWordModel

def process_page(img_path):
    
    filename = Path(img_path).name
    label = os.path.splitext(filename)[0]

    image = cv2.imread(img_path)

    configs = BaseModelConfigs.load("Models/handwriting_recognition/202312061746/configs.yaml")
    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
    df = pd.read_csv("Models/handwriting_recognition/202312061746/val.csv").values.tolist()
    accum_cer = []

    prediction_text = model.predict(image)

    cer = get_cer(prediction_text, label)
    print(f"Image: {image}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")

    accum_cer.append(cer)

    return cer, np.average(accum_cer), label, prediction_text, image
    

# define gradio interface
gr.Interface(fn=process_page,
             inputs=[gr.Image(type='filepath')],
             outputs=[gr.Textbox(label='CER'),gr.Textbox(label='Average CER'),gr.Textbox(label='Label'),gr.Textbox(label='Read Text'), gr.Image()],
             allow_flagging='never',
             title='Detect and Read Handwritten Words',
             theme=gr.themes.Monochrome()).launch()