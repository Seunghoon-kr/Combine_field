import numpy as np
import glob
import cv2

isDragging = False
x0, y0, w, h = -1, -1, -1, -1
blue, red = (255, 0, 0), (0, 0, 255)
 
def onMouse(event, x, y, flags, param):
    global isDragging, x0, y0, img
    
    if event == cv2.EVENT_LBUTTONDOWN:
        isDragging = True
        x0 = x
        y0 = y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if isDragging:
            img_draw = img.copy()
            cv2.rectangle(img_draw, (x0, y0), (x, y), blue, 2)
            cv2.imshow('img', img_draw)
    elif event == cv2.EVENT_LBUTTONUP:
        if isDragging:
            isDragging = False
            w = x - x0
            h = y - y0
            if w > 0 and h > 0:
                img_draw = img.copy()
                cv2.rectangle(img_draw, (x0, y0), (x, y), red, 2)
                cv2.imshow('img', img_draw)
                roi = img[y0:y0+h, x0:x0+w]
                cv2.imshow('cropped', roi)

                print(roi.shape)
                cv2.moveWindow('cropped', 0, 0)

                black_img = np.zeros((i_h, i_w,3))
                black_img[y0:y, x0:x, :] = roi
                b_img = black_img
                cv2.imwrite('./image/class/{}_or_{}.jpg'.format(image_num, x0), b_img)
                b_img = cv2.flip(b_img,1)
                cv2.imwrite('./image/class/{}_fl_{}.jpg'.format(image_num, x0), b_img)
                #count += 1

                
            else:
                cv2.imshow('img', img)
                print('drag should start from left-top side')


image_num = 1
slide = 1200

path = glob.glob("image/raw_img/*.jpg")
cv_img = []
print(path)
for img in path:
    print('_____________')
    img = cv2.imread(img)
    #img.append(img)
    img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
    i_w,i_h,i_ch = img.shape
    #count = 1
    cv2.imshow('img', img)
    cv2.setMouseCallback('img', onMouse)
    cv2.waitKey()
    cv2.destroyAllWindows()
    image_num +=1
    
