import cv2
import tensorflow as tf
import numpy as np
import preprocessing as prep

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, img = cap.read()
    img = prep.greenbyCOM(img)

    # Add batch dimension and preprocess if needed
    img = np.expand_dims(img, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Print prediction
    pred = np.argmax(output)
    print(pred)

    # Display webcam image
    cv2.imshow('Webcam', img[0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()