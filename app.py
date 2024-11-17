# import streamlit as st
# import pandas as pd
# import numpy as np
# from tempfile import NamedTemporaryFile
# import glob
# import re
# import os
# from pred0 import predict1, predict0

# # Title and Description
# st.title('Road Quality Assessment')
# st.markdown('Model to detect faults and check the quality of roads')

# # Select Media Type
# opt0 = st.selectbox(
#     "Select Media Type",
#     ("Image", "Video"),
#     index=0,
#     help="Choose the type of media you want to upload."
# )

# if opt0 == 'Image':
#     imgs = st.file_uploader('Upload Road images:', type=['jpg'], accept_multiple_files=False)
#     if imgs is not None:
#         st.image(imgs)

# if opt0 == 'Video':
#     imgs = st.file_uploader('Upload Road videos:', type=['avi'], accept_multiple_files=False)
#     if imgs is not None:
#         st.video(imgs)

# # Select Model
# option = st.selectbox(
#     "Select Model",
#     ("Model1", "InstanceModel"),
#     index=0,
#     help="Choose the model for road quality assessment."
# )

# st.write('Selected:', option)

# # Helper Function to Extract Numerical Part from Folder Names
# def extract_number(folder_name):
#     match = re.search(r'\d+', folder_name)
#     if match:
#         return int(match.group())
#     else:
#         return 0

# # Model Paths
# if option == 'Model1':
#     model = 'C:/Users/lenovo/Downloads/SEM6EDI/SEM6EDI/DeepDrive/DATASET_W_w.pt'
#     model0 = 'C:/Users/lenovo/Downloads/SEM6EDI/SEM6EDI/DeepDrive/DATASET_W_w.pt'
# else:
#     model0 = '/home/eshan/Eshan/Study/TY/EDI/sem6Update/ML/Runs/segment/train3/weights/best.pt'

# # Button to Run Model
# if st.button('Run Model'):
#     if imgs is not None:
#         # Save uploaded file temporarily
#         with NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
#             temp.write(imgs.getvalue())
#             temp.flush()  # Ensure the file is written to disk
#             os.chmod(temp.name, 0o644)  # Set correct permissions
            
#             # Predict using the models
#             roadArea, pts = predict1(model, temp.name)
#             cn, path, score, areas = predict0(model0, temp.name, roadArea, pts)
            
#             # Display Results
#             st.write("Class Names:", cn)
#             st.write(f"Road Quality Score: {score}")
            
#             # Remove temporary file after use
#             os.remove(temp.name)

#         # Process Results Folders
#         folders = glob.glob('pred/images*')
#         folders = sorted(folders, key=extract_number)
#         folders1 = glob.glob('pred/RoadImages*')
#         folders1 = sorted(folders1, key=extract_number)
        
#         # Extract numerical indices
#         num = int(re.findall(r'\d+', folders[-1])[0])
#         num1 = int(re.findall(r'\d+', folders1[-1])[0])
        
#         st.write("Latest Folder Index for Images:", num)
#         st.write("Latest Folder Index for Road Images:", num1)
        
#         # Display Predicted Images
#         st.image(f'/home/eshan/Eshan/Study/TY/EDI/sem6Update/ML/pred/RoadImages{num1}/{path.split("/")[-1]}')
#         st.image(f'/home/eshan/Eshan/Study/TY/EDI/sem6Update/ML/pred/images{num}/{path.split("/")[-1]}')
#     else:
#         st.error("Please upload an image or video to proceed!")


import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO  # Assuming YOLO is the model being used

# Load the pre-trained model
model = YOLO(r'C:\Users\lenovo\Downloads\SEM6EDI\SEM6EDI\DeepDrive\DATASET_W_w.pt')  # Replace with your model's path

# Streamlit App
st.title("Road Quality Detection")

# File Upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display the input image on the left
    st.subheader("Input and Output Images")
    col1, col2 = st.columns(2)
    col1.image(image, caption="Uploaded Image", channels="BGR", use_column_width=True)

    # Run the model on the uploaded image
    results = model.predict(image)

    # Extract pothole details (assumes results[0].boxes contains bounding boxes)
    pothole_areas = []
    image_area = image.shape[0] * image.shape[1]
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
        width = x2 - x1
        height = y2 - y1
        pothole_areas.append(width * height)

    # Calculate total pothole area
    total_pothole_area = sum(pothole_areas)

    # Calculate road quality score
    scaling_factor = 500  # Tune this value based on sensitivity
    normalized_impact = total_pothole_area / image_area
    road_quality_score = max(0, 100 - (normalized_impact * scaling_factor))  # Ensure score stays non-negative

    # Draw the detected results on the image
    output_image = results[0].plot()  # Assumes `plot` draws boxes/annotations on the image

    # Display the output image on the right
    col2.image(output_image, caption="Processed Output Image", channels="BGR", use_column_width=True)

    # Display the road quality score
    st.subheader("Road Quality Score")
    st.write(f"**Average Road Quality Score**: {road_quality_score:.2f}")
