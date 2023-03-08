import mediapipe as mp
import cv2
import csv
import os
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils  # Drawing helpers
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
font = cv2.FONT_HERSHEY_SIMPLEX

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return round(angle, 4)

def findDistance(x1, y1, x2, y2):
    dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return dist


cap = cv2.VideoCapture(0)


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5,model_complexity=2) as Pose:
    while cap.isOpened():
        ret, frame = cap.read()
        h, w = frame.shape[:2]

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make Detections
        body_points = Pose.process(image)
        lm = body_points.pose_landmarks
        lmPose = mp_pose.PoseLandmark
        # print(len(lm.landmark[0:10]))

        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        try:

            # select = ((lm.landmark[11:17]+lm.landmark[23:29]))
            # print((select))
            l_shldrdis_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldrdis_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
            r_shldrdis_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldrdis_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)


            #stand = findDistance(l_shldrdis_x,l_shldrdis_y,r_shldrdis_x,r_shldrdis_y)

            #if 82 < stand < 98:

             #   cv2.line(image, (450, 100), (450, 500), (0, 255, 0), 4)
              #  cv2.line(image, (180, 100), (180, 500), (0, 255, 0), 4)
            #else:

             #   cv2.line(image, (180, 100), (180, 500), (0, 0, 255), 4)
              #  cv2.line(image, (450, 100), (450, 500), (0, 0, 255), 4)

            l_shldr = (int(lm.landmark[lmPose.LEFT_SHOULDER].x * w), int(lm.landmark[lmPose.LEFT_SHOULDER].y * h))
            l_elbow = (int(lm.landmark[lmPose.LEFT_ELBOW].x * w), int(lm.landmark[lmPose.LEFT_ELBOW].y * h))
            l_wrist = (int(lm.landmark[lmPose.LEFT_WRIST].x * w), int(lm.landmark[lmPose.LEFT_WRIST].y * h))
            r_shldr = (int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w), int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h))
            r_elbow = (int(lm.landmark[lmPose.RIGHT_ELBOW].x * w), int(lm.landmark[lmPose.RIGHT_ELBOW].y * h))
            r_wrist = (int(lm.landmark[lmPose.RIGHT_WRIST].x * w), int(lm.landmark[lmPose.RIGHT_WRIST].y * h))

            r_hip = (int(lm.landmark[lmPose.RIGHT_HIP].x * w), int(lm.landmark[lmPose.RIGHT_HIP].y * h))
            r_knee = (int(lm.landmark[lmPose.RIGHT_KNEE].x * w), int(lm.landmark[lmPose.RIGHT_KNEE].y * h))
            r_ankle = (int(lm.landmark[lmPose.RIGHT_ANKLE].x * w), int(lm.landmark[lmPose.RIGHT_ANKLE].y * h))
            l_hip = (int(lm.landmark[lmPose.LEFT_HIP].x * w), int(lm.landmark[lmPose.LEFT_HIP].y * h))
            l_knee = (int(lm.landmark[lmPose.LEFT_KNEE].x * w), int(lm.landmark[lmPose.LEFT_KNEE].y * h))
            l_ankle = (int(lm.landmark[lmPose.LEFT_ANKLE].x * w), int(lm.landmark[lmPose.LEFT_ANKLE].y * h))
            nose = (int(lm.landmark[lmPose.NOSE].x * w), int(lm.landmark[lmPose.NOSE].y * h))


            cv2.circle(image, (l_elbow), 7, (127, 255, 0), 6)
            cv2.circle(image, (l_wrist), 7, (127, 255, 0), 6)
            cv2.circle(image, (l_shldr), 7, (127, 255, 0), 6)
            cv2.circle(image, (r_elbow), 7, (127, 255, 0), 6)
            cv2.circle(image, (r_wrist), 7, (127, 255, 0), 6)
            cv2.circle(image, (r_shldr), 7, (127, 255, 0), 6)
            cv2.circle(image, (l_hip), 7, (127, 255, 0), 6)
            cv2.circle(image, (l_knee), 7, (127, 255, 0), 6)
            cv2.circle(image, (l_ankle), 7, (127, 255, 0), 6)
            cv2.circle(image, (r_ankle), 7, (127, 255, 0), 6)
            cv2.circle(image, (r_hip), 7, (127, 255, 0), 6)
            cv2.circle(image, (r_knee), 7, (127, 255, 0), 6)
            cv2.circle(image, (nose), 7, (0, 255, 0), 6)
            # cv2.circle(image, (150,50), 5, (127, 20, 0), 6)

            cv2.line(image, (l_shldr), (l_elbow), (0, 0, 255), 4)
            cv2.line(image, (r_shldr), (r_elbow), (0, 0, 255), 4)
            cv2.line(image, (l_elbow), (l_wrist), (0, 0, 255), 4)
            cv2.line(image, (r_elbow), (r_wrist), (0, 0, 255), 4)
            cv2.line(image, (l_hip), (l_knee), (0, 0, 255), 4)
            cv2.line(image, (l_knee), (l_ankle), (0, 0, 255), 4)
            cv2.line(image, (r_hip), (r_knee), (0, 0, 255), 4)
            cv2.line(image, (r_knee), (r_ankle), (0, 0, 255), 4)
            cv2.line(image, (l_shldr), (l_hip), (0, 0, 255), 4)
            cv2.line(image, (r_shldr), (r_hip), (0, 0, 255), 4)
            cv2.line(image, (r_hip), (l_hip), (0, 0, 255), 4)
            cv2.line(image, (r_shldr), (l_shldr), (0, 0, 255), 4)
            #cv2.line(image, (180,100), (180,500), (0, 0, 255), 4)
            #cv2.line(image, (450, 100), (450, 500), (0, 0, 255), 4)
           # cv2.putText(image, str(fps),
                  #      (10, 100),
                  #      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        except:
            pass

        cv2.imshow('AI Fitness', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
select = (int(lm.landmark[lmPose.LEFT_SHOULDER].x * w), int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)), (
    int(lm.landmark[lmPose.LEFT_ELBOW].x * w), int(lm.landmark[lmPose.LEFT_ELBOW].y * h)), \
         (int(lm.landmark[lmPose.LEFT_WRIST].x * w), int(lm.landmark[lmPose.LEFT_WRIST].y * h)), (
             int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w), int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)), \
         (int(lm.landmark[lmPose.RIGHT_ELBOW].x * w), int(lm.landmark[lmPose.RIGHT_ELBOW].y * h)), (
             int(lm.landmark[lmPose.RIGHT_WRIST].x * w), int(lm.landmark[lmPose.RIGHT_WRIST].y * h)), \
         (int(lm.landmark[lmPose.RIGHT_HIP].x * w), int(lm.landmark[lmPose.RIGHT_HIP].y * h)), \
         (int(lm.landmark[lmPose.RIGHT_KNEE].x * w), int(lm.landmark[lmPose.RIGHT_KNEE].y * h)), \
         (int(lm.landmark[lmPose.RIGHT_ANKLE].x * w), int(lm.landmark[lmPose.RIGHT_ANKLE].y * h)), \
         (int(lm.landmark[lmPose.LEFT_HIP].x * w), int(lm.landmark[lmPose.LEFT_HIP].y * h)), \
         (int(lm.landmark[lmPose.LEFT_KNEE].x * w), int(lm.landmark[lmPose.LEFT_KNEE].y * h)), \
         (int(lm.landmark[lmPose.LEFT_ANKLE].x * w), int(lm.landmark[lmPose.LEFT_ANKLE].y * h)), \
         (int(lm.landmark[lmPose.NOSE].x * w), int(lm.landmark[lmPose.NOSE].y * h))
# print((select))
tupel_list = list(select)
# print(tupel_list)
lndmrk = []

for t in tupel_list:
    for x in t:
        lndmrk.append(x)
        num = len(lndmrk)
print((num))

landmarks = ['class']
for val in range(13):
    landmarks += ['x{}'.format(val),'y{}'.format(val)]
print(landmarks)

#with open('wwwwwdataset.csv', mode='w', newline='') as f:
    #csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
   # csv_writer.writerow(landmarks)
    
