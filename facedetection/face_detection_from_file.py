#
# Python 2.6 and OpenCV 2.0 face detection
#
# Detects face(s) visible to a webcam using a bright green border
#
# This code is very heavily based on code from:
# http://blog.jozilla.net/2008/06/27/fun-with-python-opencv-and-face-detection/
#
# The code has been updated to work with OpenCV 2.0 and its new python bindings.
#
# This code requires OpenCV 2.0 - very very good instructions available here:
# http://subsumption.blogspot.com/2010/02/these-are-instructions-i-used-to.html
#
# Finally, you need the Haar casade file:
# http://eclecti.cc/files/2008/03/haarcascade_frontalface_alt.xml
#

import sys
import cv


def detect(image):
    image_size = cv.GetSize(image)

    # create grayscale version
    grayscale = cv.CreateImage(image_size, 8, 1)
    cv.CvtColor(image, grayscale, cv.CV_BGR2GRAY)

    # create storage
    storage = cv.CreateMemStorage(0)

    # equalize histogram
    cv.EqualizeHist(grayscale, grayscale)

    # show processed image
    cv.ShowImage('Processed', grayscale)

    # detect objects
    cascade = cv.Load('haarcascade_frontalface_alt.xml')
    faces = cv.HaarDetectObjects(grayscale, cascade, storage, 1.2, 2, cv.CV_HAAR_DO_CANNY_PRUNING)

    if faces:
        for i in faces:
            cv.Rectangle(image,
                         (i[0][0], i[0][1]),
                         (i[0][0] + i[0][2], i[0][1] + i[0][3]),
                         (0, 255, 0),
                         3,
                         8,
                         0)


if __name__ == "__main__":
    print "Press ESC to exit ..."

    # create windows
    cv.NamedWindow('Raw', cv.CV_WINDOW_AUTOSIZE)
    cv.NamedWindow('Processed', cv.CV_WINDOW_AUTOSIZE)

    while 1:
        # do forever

        # Load file from argv list or alternatively, 
        # load 'demo.jpg'
        filename = "demo.jpg"
        if len(sys.argv) == 2:
            filename = sys.argv[1]

        frame = cv.LoadImage(filename)

        # mirror
        cv.Flip(frame, None, 1)

        # face detection
        detect(frame)

        # display webcam image
        cv.ShowImage('Raw', frame)

        # handle events
        k = cv.WaitKey(10)

        if k == 0x1b: # ESC
            print 'ESC pressed. Exiting ...'
            break
