import numpy as np
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
                #black backgraound image에 저장
                
                roi_h, roi_w, roi_ch = roi.shape
                w_slide_num = (i_w - roi_w) // slide + 1
                h_slide_num = (i_h - roi_h) // slide + 1
                print(w_slide_num)
                print(h_slide_num)
                count = 1

                for i in range(h_slide_num):
                    for j in range(w_slide_num):
                        #print(i)
                        black_img = np.zeros((i_h, i_w,3))
                        black_img[i*slide:i*slide + roi_h, j*slide:j*slide + roi_w, :] = roi
                        
                        b_img = black_img
                        #b_img = cv2.resize(black_img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
                        cv2.imwrite('./image/class/{}_or_{}_{}.jpg'.format(image_num, count, roi_w), b_img)
                        b_img = cv2.flip(b_img,1)
                        cv2.imwrite('./image/class/{}_fl_{}_{}.jpg'.format(image_num, count, roi_w), b_img)
                        count += 1
                        j += 1
                    i += 1

                
                #cv2.imwrite('./image/class/cropped.png', roi)
            else:
                cv2.imshow('img', img)
                print('drag should start from left-top side')


image_num = 1
slide = 500
img = cv2.imread('image/raw_img/1.jpg')
#i_w,i_h,i_ch = img.shape
i_w,i_h = 4032, 3024
#i_w,i_h = 3024, 4032
cv2.imshow('img', img)
cv2.setMouseCallback('img', onMouse)
cv2.waitKey()
cv2.destroyAllWindows()





'''
black_image = Image.fromarray(black_image, 'RGB')

black_image.show()
'''

