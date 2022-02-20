from keras.models import model_from_json
import tensorflow as tf
import gradio as gr
import numpy as np
import cv2 as cv2

json_file = open('./model/model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("./model/model1.h5")

labels=["Octagon","Triangle","Circle Prohibitory","Circle","Rhombus"]

def fill(img):
  cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  for c in cnts:
      cv2.drawContours(img,[c], 0, (255,255,255), -1)
  return img

def save(img):
    width = 32
    height = 32
    dim = (width, height)
    resized=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return cv2.imwrite('./image_temp.png',resized)

def shape_recognition():
    image=cv2.imread('./image_temp.png')
    image_array= np.expand_dims(image, axis=0)
    predictions=model.predict(image_array)
    score = tf.nn.softmax(predictions[0])
    return {labels[i]: float(score[i]) for i in range(len(labels))}
    
def convert(image):
    result = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower1 = np.array([0, 100, 20])
    upper1 = np.array([10, 255, 255])
    
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([160,100,20])
    upper2 = np.array([179,255,255]) 
    
    lower_blue_1 = np.array([112,50,50])
    upper_blue_1 = np.array([130,255,255])
    lower_blue_2 = np.array([96, 80, 2])
    upper_blue_2 = np.array([126, 255, 255])

    lower_red_mask = cv2.inRange(image, lower1, upper1)
    upper_red_mask = cv2.inRange(image, lower2, upper2)

    lower_blue_mask =cv2.inRange(image, lower_blue_1, upper_blue_1)
    upper_blue_mask =cv2.inRange(image, lower_blue_2, upper_blue_2)

    red_full_mask = lower_red_mask + upper_red_mask
    blue_full_mask = lower_blue_mask + upper_blue_mask 
    blue_only= cv2.bitwise_and(result, result, mask = blue_full_mask)
    
    # full_mask=blue_full_mask+red_full_mask
    # result = cv2.bitwise_and(result, result, mask=full_mask)
    if np.count_nonzero(red_full_mask) < np.count_nonzero(blue_full_mask):
        if np.count_nonzero(blue_only==[78,6,0])>np.count_nonzero(blue_only):
            filled=fill(red_full_mask)

            save(filled)
            bg_removed = cv2.bitwise_and(result, result, mask = filled)
            return shape_recognition(),bg_removed,"Red"
        elif np.count_nonzero(blue_only==[78,6,0])<np.count_nonzero(blue_only):
            filled=fill(blue_full_mask)
            save(filled)
            bg_removed = cv2.bitwise_and(result, result, mask = filled)
            return shape_recognition(),bg_removed,"Blue"
    elif np.count_nonzero(blue_full_mask) < np.count_nonzero(red_full_mask):
        filled=fill(red_full_mask)
        save(filled)
        bg_removed = cv2.bitwise_and(result, result, mask = filled)
        return shape_recognition(),bg_removed,"Red"
    else:
        return result, "undefined"




iface=gr.Interface(convert,
    inputs=gr.inputs.Image(label="Upload an Image"),
    outputs=[gr.outputs.Label(num_top_classes=1, label="Shape"),
    gr.outputs.Image(label="Removed Background Image"),
    gr.outputs.Label(label="Color"),
    ],
    live=True,
    title="Traffic Sign Shape & Color Detection",
    examples=['./examples/1.jpg','./examples/2.jpg', './examples/got.jpg', './examples/goti.jpg', './examples/images.jpg'],
    description='This is my Thesis Project'
)

iface.launch(debug=True,inbrowser=True)