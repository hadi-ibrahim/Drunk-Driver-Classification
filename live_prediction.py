import os
import glob
import cv2
import numpy as np
from collections import deque


from tensorflow.keras.models import load_model

model = load_model('./best_model')
classes = [i.split(os.path.sep)[1] for i in glob.glob('./Data/*')]


mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
queue = deque(maxlen=128)

# start running cell from here till the end after loading the model
cap = cv2.VideoCapture(0)

writer = None
(W, H) = (None, None)
buffer = []
old_label= None
while(True):
    
    while (len(buffer) < 5):
        grabbed, frame = cap.read()
    
        if not grabbed:
            break
        if(old_label is not None):
            text = "state: {}".format(old_label)
            cv2.putText(frame, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,1.25, (0, 255, 0), 5)
        cv2.imshow("Main detection", frame)
        key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (112, 112)).astype("float32")
        buffer.append(frame)
    
    if key == ord("q"):
        break
        
    print("Predicting sequence")
    try:
        preds = model.predict(np.expand_dims(np.asarray(buffer),axis=0))
        buffer.clear()
	# perform prediction averaging over the current history of
        queue.append(preds)
    except:
        print("crashed")
        break
    
    results = np.array(queue).mean(axis=0)
    print (results)
    i = np.argmax(results)
    print (i)
    label = classes[i]
    old_label=label
    
    buffer.clear()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()