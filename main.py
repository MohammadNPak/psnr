import numpy as np
import cv2

print(np.__version__)
print(cv2.version.opencv_version)

cap = cv2.VideoCapture(r'E:\movie\harryPotter\video1.mkv')
cap2 = cv2.VideoCapture(r'E:\movie\harryPotter\video1_encoded.mkv')
frame_counter = 1
while cap.isOpened():
    ret1, frame1 = cap.read()

    ret2, frame2 = cap.read()

    if (cv2.waitKey() & 0xFF == ord('q')) or not ret1 or not ret2:
        break

    diff = (np.square(frame1.astype('int32') - frame2.astype('int32'))).astype('uint8')
    frame1_yuv = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV)
    frame2_yuv = cv2.cvtColor(frame2, cv2.COLOR_BGR2YUV)
    diff_yuv = (np.square(frame1_yuv.astype('int32') - frame2_yuv.astype('int32'))).astype('uint8')

    mse = np.sum(diff, axis=(0, 1)) / (diff.shape[0] * diff.shape[1])
    mse_yuv = np.sum(diff_yuv, axis=(0, 1)) / (diff.shape[0] * diff.shape[1])
    print('rgb', frame_counter, mse)
    print('yuv', frame_counter, mse_yuv)
    frame_counter += 1
    cv2.imshow("frame1", cv2.resize(frame1, (0, 0), None, 0.5, 0.5))
    cv2.imshow("frame2", cv2.resize(frame2, (0, 0), None, 0.5, 0.5, None))
    cv2.imshow("diff", cv2.resize(diff, (0, 0), None, 0.5, 0.5, None))

cv2.destroyAllWindows()
