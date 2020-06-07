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

# change name of the video or directory for predictions of different videos
# example : cv2.VideoCapture('./Data/drunk/drunk_driving_3_Trim4.mp4') for drunk videos
cap = cv2.VideoCapture('./Data/normal/normal_driving_3_Trim4.mp4')

writer = None
(W, H) = (None, None)
buffer = []
final_video = []
while(True):
    # read the next frame from the file
    while(len(buffer)<5):
        (grabbed, frame) = cap.read()
	# if the frame was not grabbed, then we have reached the end
	# of the stream
        if not grabbed:
            break
	# if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        
    # to show output when done 
        output = frame.copy()
        final_video.append(output)
    # clone the output frame, then convert it from BGR to RGB
	# ordering, resize the frame to a fixed 224x224, and then
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (112, 112)).astype("float32")
        buffer.append(frame)
    
    # make predictions on the frame and then update the predictions
    print("Predicting sequence")
    try:
        preds = model.predict(np.expand_dims(np.asarray(buffer),axis=0))
        buffer.clear()
	# perform prediction averaging over the current history of
        queue.append(preds)
    except:
        break
    
	# previous predictions
    results = np.array(queue).mean(axis=0)
    print (results)
    i = np.argmax(results)
    print (i)
    label = classes[i]
    
    # draw the activity on the output frame
    text = "state: {}".format(label)
    for f in final_video:
        cv2.putText(f, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
		1.25, (0, 255, 0), 5)
	# check if the video writer is None
        if writer is None:
		# initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter("prediction.avi", fourcc, 30,(W, H), True)
	# write the output frame to disk
        writer.write(f)
    final_video.clear()

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()