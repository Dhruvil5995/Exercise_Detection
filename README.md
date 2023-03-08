# GYM Exercise Detection using Human Pose Estimation

This project uses human pose estimation techniques to detect gym exercises, such as pushups, squats, bicep curls, and shoulder exercises. The system is built using the MediaPipe library and generates a skeleton model with 33 key points on the human body.
Each landmark or keypoint consists of four values: X, Y, Z, and visibility, resulting in 132 values for the entire skeleton model.

# Skeleton model with 33 landmarks

![image](https://user-images.githubusercontent.com/64741151/223779504-c8d25977-02c5-4a52-aebc-9269fb60a3fa.png)

# Skeleton model with only important landmarks

![image](https://user-images.githubusercontent.com/64741151/223779861-1540b613-82df-4036-8b91-aa903760ed90.png)

Initially, the system was built using 33 key points, but found that this resulted in some keypoints overlapping on each other due to the large number of points on the face, hands, and feet. This made it difficult to accurately track the user's posture and movements during exercise.
To address this issue, i reduced the number of keypoints to 13, focusing on the most important points for exercise detection.


# Data Collection

As an example, place a skeleton model with  key points in the desired posture in front 
of the webcam for a set amount of time. Posture landmark values are saved in a CSV file 
during this time.
To create a good dataset for the machine learning model, the X and Y values of each landmark are multiplied by the image width and height instead of taking each landmark's four points (X, Y, Z, and visibility).

# Result

The system can be used for real-time exercise detection, providing instant feedback on the user's posture and performance. It can also be used for data analysis and machine learning, allowing researchers to study human movements and develop new exercise detection algorithms.

