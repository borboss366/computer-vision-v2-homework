import cv2
import numpy as np
import time

def create_tracker():
    return cv2.TrackerKCF_create()

