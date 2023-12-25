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
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from inferenceModel import ImageToWordModel

def process_page(img_path):
    
    filename = Path(img_path).name
    label = os.path.splitext(filename)[0]

    image = cv2.imread(img_path)

    configs = BaseModelConfigs.load("Models/sentence_recognition/202312071202/configs.yaml")

    model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

    accum_cer, accum_wer = [], []

    image = cv2.imread(img_path)

    prediction_text = model.predict(image).strip()

    cer = get_cer(prediction_text, label)
    wer = get_wer(prediction_text, label)
 

    accum_cer.append(cer)
    accum_wer.append(wer)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return cer, np.average(accum_cer),wer,np.average(accum_wer) ,label, prediction_text, image
    

# define gradio interface
gr.Interface(fn=process_page,
             inputs=[gr.Image(label='Input image', type='filepath')],
             outputs=[gr.Textbox(label='CER'),gr.Textbox(label='Average CER'),gr.Textbox(label='WER'),gr.Textbox(label='Average WER'),gr.Textbox(label='Label'),gr.Textbox(label='Read Text'), gr.Image(show_label=1)],
             allow_flagging='never',
             title='Detect and Read Handwritten Words',
             theme=gr.themes.Monochrome()).launch()