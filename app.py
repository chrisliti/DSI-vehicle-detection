## Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


import streamlit as st

import tensorflow.keras
from tensorflow.keras.models import load_model
import pickle

#Loading the Model

truck_model = load_model('best_model.h5')

#Name of Classes
a_file = open("vehicle_dict.pkl", "rb")
ref = pickle.load(a_file)

a_file.close()

#Setting Title of App
st.title("Vehicle Classification")
st.markdown("Upload vehicle image")


#Uploading the dog image
vehicle_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict')

#On predict button click
if submit:


    if vehicle_image is not None:

         # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(vehicle_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (256,256))
    
        #Convert image to 4 Dimension
        opencv_image.shape = (1,256,256,3)

        #Make Prediction
        pred = np.argmax(truck_model.predict(opencv_image))
        prediction = ref[pred]
        st.markdown(str("Image belongs to "+prediction))