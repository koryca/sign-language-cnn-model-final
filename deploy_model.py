import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import time

st.title("Image Classification")

def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Submit")
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                image = Image.open(file_uploaded)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Predicted number is {}'.format(predictions))
                # st.write(predictions)
         

def predict(image):
    classifier_model = "model_signLang_numbers2.hdf5" #check model file
    model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    test_image = image.resize((64,64))
    test_image = preprocessing.image.img_to_array(test_image)
    #test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    #plt.imshow(test_image)
    class_names = ['0','1','2','3','4','5','6','7','8','9']
    predictions = model.predict(test_image)
    print(predictions[0])
    
    # result = f"{class_names[np.argmax(tf.nn.softmax(predictions[0]))]}" 
    scores = tf.nn.softmax(predictions[0]) #check this part
    scores = scores.numpy()
    print(scores)
    
    result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } percent confidence."
    return result


if __name__ == "__main__":
    main()