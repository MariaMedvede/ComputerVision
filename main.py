# bin/setupvar.bat
#project in openvino
#pip install opencv-python
#https://download.01.org/opencv/2019/open_model_zoo/R1/models_bin/emotions-recognition-retail-0003/
#https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_raspbian.html

from openvino.inference_engine import IENetwork, IECore
#from Pillow import Image
net = IENetwork('C:\\Users\\maria\\Downloads\\face-detection-adas-0001.xml', 'C:\\Users\\maria\\Downloads\\face-detection-adas-0001.bin')

ie = IECore()
ie.add_extension('cpu_extension_avx2.dll', "CPU")
exec_net = ie.load_network(net, 'CPU')

#"C:\Users\maria\Downloads\face-detection-adas-0001.xml"
import cv2

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()

    input = cv2.dnn.blobFromImage(frame, size=(672, 384))
    outputs = exec_net.infer(inputs={'data': input})
    cv2.imwrite('test.png', frame)


    for detection in outputs['detection_out'].reshape(-1, 7):
        confidence = float(detection[2])
        xmin = int(detection[3] * frame.shape[1])
        ymin = int(detection[4] * frame.shape[0])
        xmax = int(detection[5] * frame.shape[1])
        ymax = int(detection[6] * frame.shape[0])
        if confidence > 0.5:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
            face_img = frame[ymin:ymax, xmin:xmax]

            #area = (xmin, ymin, xmax, ymax)
            #img = Image.open("test.png")
            #cropped_image=img.crop(area)
            cv2.imwrite("las.png", face_img)
            #img = input
            #roi_color = img[ymin:ymin + ymax, xmin:xmin + xmax]



    # Save the frame to an image file.
    cv2.imshow('out.png', frame)


    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
vc.release()
cv2.destroyWindow("preview")

netemot = IENetwork('model.xml', 'model.bin')
iee = IECore()
iee.add_extension('cpu_extension_avx2.dll', "CPU")
exec_net = iee.load_network(netemot, 'CPU')
img = cv2.resize(face_img,(64,64),3)
outputs2 = exec_net.infer(inputs={'data': img.reshape(1, 3, 64, 64)})
print(outputs2)