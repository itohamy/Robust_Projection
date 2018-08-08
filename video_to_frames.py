import cv2


def extractImages(pathIn, pathOut):
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
      success,image = vidcap.read()
      cv2.imwrite( pathOut + "frame%d.jpg" % count, image)     # save frame as JPEG file
      count += 1
    if count > 0:
        print('%d images extracted successfully' % count)
    else:
        print('Images extraction failed.')


# if __name__=="__main__":
#     print("aba")
#     extractImages("spity.mp4", "frames/")