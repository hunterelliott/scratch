import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from screeninfo import get_monitors
import torch
from torchvision.transforms import Normalize


CAM_DEVICE = 0
WIN_NAME = 'projector'
MON_NUM = 0
RESOLUTION = (1920,1080)  # WxH - 1080p

cap = cv.VideoCapture(CAM_DEVICE)
# cap.open(0, apiPreference=cv.CAP_V4L2)
cap.set(cv.CAP_PROP_FRAME_WIDTH,RESOLUTION[0])
cap.set(cv.CAP_PROP_FRAME_HEIGHT,RESOLUTION[1])


cv.namedWindow(WIN_NAME, cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty(WIN_NAME, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
cv.moveWindow(WIN_NAME, get_monitors()[MON_NUM].x, get_monitors()[MON_NUM].y)

assert cap.isOpened()
assert torch.cuda.is_available()

device = torch.device("cuda")
model =  torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=3, init_features=32, pretrained=False).to(device)



while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break


    frame_rs = cv.resize(frame,(256,256))

    model_in = torch.tensor(frame_rs).float().movedim(2, 0).unsqueeze(0).to(device)
    in_mean = torch.mean(model_in,(2,3), keepdim=True)
    in_std = torch.std(model_in,(2,3), keepdim=True)
    model_in = model_in - in_mean
    model_in = model_in / (in_std + 1e-5)

    model_out = model(model_in).detach()

    model_out = model_out * in_std
    model_out = model_out + in_mean
    model_out = model_out.cpu().squeeze().movedim(0,2).numpy().astype(np.uint8)

    out_rs = cv.resize(model_out, RESOLUTION)

    # Display the resulting frame
    # cv.imshow(WIN_NAME, frame)
    cv.imshow(WIN_NAME, out_rs)


    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

j=1



