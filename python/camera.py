import ctypes
import time
import cv2
import numpy as np
import faulthandler

faulthandler.enable()

libname = '../cpp/build/host/libbarcode_segmentation_lib.so'
clib = ctypes.CDLL(libname)

run_mdd_drt = clib.run_mdd_drt

run_mdd_drt.restype = ctypes.POINTER(ctypes.c_uint8 * 512 * 512 * 3)
run_mdd_drt.argtypes = [np.ctypeslib.ndpointer(dtype=np.dtype('uint8'), shape=(1024, 1024)),
                        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]

capture = cv2.VideoCapture(0)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FPS, 30)
capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
number_of_frames_elapsed = 0
last_time = time.time()
while capture.isOpened():
    ret, frame = capture.read()
    h, w, _ = frame.shape
    crop = (w - h) // 2
    frame = frame[:, crop:-crop]
    frame = frame[:, :, 0]
    cv2.imshow('Input', frame)
    frame = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_LINEAR)

    number_of_frames_elapsed += 1
    if number_of_frames_elapsed == 50:
        time_per_frame = time.time() - last_time
        time_per_frame /= 100
        print(f'FPS: {1 / time_per_frame}')
        last_time = time.time()
        number_of_frames_elapsed = 0

    in_img = np.double(frame)
    in_img -= in_img.min()
    in_img /= in_img.max()
    in_img = np.uint8(255 * in_img)
    mdd_weights = [0.05, 0.527, 0.33, 0.76, 0.84, 0.84, 1.16, 3.47]
    mdd_threshold = 1
    mdd_result = run_mdd_drt(in_img, *mdd_weights, mdd_threshold)
    mdd_result = np.array(mdd_result.contents)
    mdd_result = np.moveaxis(mdd_result, 0, 2)
    mdd_result = cv2.cvtColor(mdd_result, cv2.COLOR_RGB2BGR)
    cv2.imshow('MDD DRT', mdd_result)
    cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()
