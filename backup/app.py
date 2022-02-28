from keras.models import model_from_json
import tensorflow as tf
import gradio as gr
import numpy as np
import cv2 as cv2
from PIL import Image, ImageFilter

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

def center_of_contour(contour):
    cnts = cv2.findContours(contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])  
        return cX, cY

def center_of_image(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    moment = cv2.moments(gray_img)
    X = int(moment ["m10"] / moment["m00"])
    Y = int(moment ["m01"] / moment["m00"])
    return X,Y

def distance_of_centers(center_of_image,center_of_contour):
    dist = np.sqrt(np.sum(np.square(center_of_image - center_of_contour))) 
    return dist
def save(img):
    img= cv2.bilateralFilter(img,9,75,75)
    width = 32
    height = 32
    dim = (width, height)
    resized=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return cv2.imwrite('./image_temp.png',resized)

def shape_recognition(number):
    image=cv2.imread('./image_temp.png')
    image_array= np.expand_dims(image, axis=0)
    predictions=model.predict(image_array)
    score = tf.nn.softmax(predictions[0])
    # return {f"{number +' '+ labels[i] }": float(score[i]) for i in range(len(labels))}
    return f'Shape {str(number)}:'+" "+ f'{labels[np.argmax(score)]}'
    
def detect(image):
    shape_n=[]
    color=[]
    with tf.io.gfile.GFile('./detection_model/frozen_inference_graph.pb', 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.compat.v1.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # Read and preprocess an image.
        img = image
        cropper=img
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv2.resize(img, (300, 300))
        #inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                    feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})     
        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        roi=[]
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.8:
                x = (bbox[1] * cols) -10 #left
                y = (bbox[0] * rows) - 15 #top
                right = (bbox[3] * cols) + 10
                bottom = (bbox[2] * rows ) +10
                crop=cropper[int(y): int(bottom),int(x):int(right)]
                roi.append(crop)
                detect=cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (15, 255,100), thickness=4)
                cv2.putText(detect, f'Sign #{i+1}', (int(x), int(y-10)), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.0, (255,255,233), 3)
    
    count=1
    for i in roi:
        if len(i)>0:
            result = i.copy()
            image = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
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
            
            full_mask=blue_full_mask+red_full_mask
            # result = cv2.bitwise_and(result, result, mask=full_mask)
            if np.count_nonzero(red_full_mask) < np.count_nonzero(blue_full_mask):

                if np.count_nonzero(blue_only==[78,6,0])>np.count_nonzero(blue_only):

                    if np.count_nonzero(red_full_mask)>1000:
                        filled=fill(red_full_mask)
                        save(filled)
                        #bg_removed = cv2.bitwise_and(result, result, mask = filled)
                        shape_n.append(shape_recognition(count))
                        color.append(f"Color{count}: Red")
                    else:
                        shape_n.append(f"Shape{count}: Undefined")
                        color.append(f"Color{count}: Undefined ")
                
                elif np.count_nonzero(red_full_mask)<np.count_nonzero(blue_full_mask):
                    """We've got problem in this block"""
                    if np.count_nonzero(red_full_mask) != 0:
                        print(np.count_nonzero(red_full_mask))
                        if np.count_nonzero(red_full_mask)>500:
                            filled=fill(red_full_mask)
                            save(filled)
                            #bg_removed = cv2.bitwise_and(result, result, mask = filled)
                            shape_n.append(shape_recognition(count))
                            color.append(f"Color{count}: REd")
                        elif np.count_nonzero(blue_full_mask)>500:
                            filled=fill(blue_full_mask)
                            save(filled)
                            #bg_removed = cv2.bitwise_and(result, result, mask = filled)
                            shape_n.append(shape_recognition(count))
                            color.append(f"Color{count}: BlUe")    
                    elif np.count_nonzero(red_full_mask)==0:
                        filled=fill(blue_full_mask)
                        save(filled)
                        #bg_removed = cv2.bitwise_and(result, result, mask = filled)
                        shape_n.append(shape_recognition(count))
                        color.append(f"Color{count}: Blue")

            elif np.count_nonzero(blue_full_mask) < np.count_nonzero(red_full_mask):
                filled=fill(red_full_mask)
                save(filled)
                #bg_removed = cv2.bitwise_and(result, result, mask = filled)
                shape_n.append(shape_recognition(count))
                color.append(f"Color{count}: Red")
        else:
            return img, "undefined", "undefined"
        
        count+=1

    return detect, ', '.join(shape_n), ', '.join(color)


iface=gr.Interface(detect,
    inputs=gr.inputs.Image(label="Upload an Image"),
    outputs=[gr.outputs.Image(label="Detected Image"),
    gr.outputs.Label(label="Shape"),
    # gr.outputs.Image(label="Removed Background Image"),
    gr.outputs.Label(label="Color")
    ],
    title="Traffic Sign Detection with Shape & Color Description",
    examples=['examples/Screenshot (57).png','./examples/1.jpg','./examples/2.jpg', './examples/got.jpg', './examples/goti.jpg', './examples/images.jpg'],
    description='This is my Thesis Project',
    theme='grass'
)

iface.launch(debug=True)