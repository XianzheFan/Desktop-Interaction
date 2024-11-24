# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import mediapipe as mp

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from track.hungarian import Hungarian, Brightness, Match_loc
from utils.sits import sit_2


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    # INFO: activate, bboxes, bright, count, cls
    INFO = []
    # to record any possible misdetect
    len_INFO = 0
    c_INFO = 0
    # t_yolo,t_sort,t_light = 0,0,0
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    frames = 0
    mark = [223,443,152,5]
    situation0 = [-1,-1] # person exists and phone on desk
    situation1 = -1 # face direction
    situation2 = [-1,0] # user appears in the middle of image, but return and sit down??
        # person id, count
    situation3 = False # group discussion, multiple faces in face detection
    count_sit3 = 0

    hands = mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) 
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    t_MP = 0
    T = 0

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('../../../FaceRecognizer/trainner/trainner.yml')
    face_detector = cv2.CascadeClassifier('../../../FaceRecognizer/models/haarcascade_frontalcatface_extended.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX
    idnum = 0
    names_FR = ['user']
    user_id = -1
    direction1,direction2 = "",""

    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            imMP = im0
            imFR = im0
            w, h = im0.shape[1], im0.shape[0]

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                Loc = []
                Cls = []
                People = [] # to compare 

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # w, h = im0.shape[1], im0.shape[0]
                   
                    # record xyxy for this frame, make assignment only based on xy
                    Loc.append([int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])])
                    Cls.append(cls)
                
                if len(INFO) != 0:
                    LOC = []
                    CLS = []
                    ID = []
                    B = []
                    device_direction = []
                    for idx, info in enumerate(INFO):
                        if info[0] != 0: # exist in last frame
                            LOC.append(info[1])
                            CLS.append(info[4])
                            ID.append(idx)
                            if (int(info[4]) == 1) | (int(info[4]) == 0):
                                loc = info[1]
                                loc_w = int((loc[0]+loc[2])/2)
                                loc_h = int((loc[1]+loc[3])/2)
                                direc1 = "left" if loc_w>400 else ("right" if loc_w<240 else "middle")
                                direc2 = "down" if (loc_h<400)&(loc_h>320) else "up"
                                # print(idx,int(info[4]),loc_w,loc_h,direc1,direc2)
                                device_direction.append([idx,direc1,direc2])
                                # if (direc1 is direction1) & ((direc2 is direction2)|(direc1 is not "middle")):
                                #     cv2.putText(im0,"face direction",(loc_w,loc_h),font,0.5,(0,0,255),1)
                                #     im0_mask = np.zeros(im0.shape, dtype=np.uint8)
                                #     cv2.rectangle(im0_mask,(loc[0],loc[1]),(loc[2],loc[3]),(255,255,0),-1)
                                #     im0 = cv2.addWeighted(im0, 1.0, im0_mask, 0.5, 0)
                                    # im0 = 0.5*im0_mask + im0
                    situation1 = -1
                    for (idx,direc1,direc2) in device_direction:
                        if (direc1 is direction1) & (situation1 == -1):
                            situation1 = idx
                        elif (direc1 is direction1) & (situation1 != -1):
                            situation1 = -1
                            break
                    if situation1 != -1:
                        loc = INFO[idx][1]
                        loc_w = int((loc[0]+loc[2])/2)
                        loc_h = int((loc[1]+loc[3])/2)
                        cv2.putText(im0,"face direction",(loc_w,loc_h),font,0.5,(0,0,255),1)
                    
                    # print("LOC: ",LOC)
                    # print("Loc: ",Loc)
                    match = [] # Loc to INFO
                    for j in range(len(Loc)):
                        match.append(0.0)
                    state = np.zeros(len(Loc)) # ON or OFF
                    row_idx, col_idx = Hungarian(LOC,Loc,CLS,Cls) #Loc to LOC, LOC to Loc
                    # print("row_idx:",row_idx,"\ncol_idx:",col_idx)
                    for j in range(len(row_idx)):
                        if row_idx[j] >= 0:
                            row_idx[j] = int(row_idx[j])
                        if col_idx[j] >= 0:
                            col_idx[j] = int(col_idx[j])    
                    for j in range(len(Loc)):
                        loc = Loc[j]
                        img_temp = imc[loc[1]:loc[3],loc[0]:loc[2]]
                        B.append(Brightness(img_temp))
                    for j in range(len(col_idx)):
                        # if row_idx[j] >= 0:
                        #     match[j] = ID[row_idx[j]]
                        if col_idx[j] >= 0:
                            if (int(INFO[ID[j]][4]) != 0) & (int(INFO[ID[j]][4]) != 1):
                                # print("Not Pad or Phone!",int(INFO[ID[j]][4]))
                                INFO[ID[j]] = [1,Loc[col_idx[j]],0,0,INFO[ID[j]][4]]
                                match[col_idx[j]] = ID[j]
                                continue
                            bright0 = INFO[ID[j]][2]
                            bright = B[col_idx[j]]
                            c0 = INFO[ID[j]][3]
                            if bright > bright0*1.02 :
                                c = c0+1 if c0 > 0 else 1
                            elif bright < bright0*0.98:
                                c = c0-1 if c0 < 0 else -1
                            else:
                                c = 0
                            if ~(Match_loc(0.1,INFO[ID[j]][1],Loc[col_idx[j]])):
                                c = 0
                            if c0 >= 3:
                                state[col_idx[j]] = 1
                                print("{0} Turn ON!!!".format(id))
                            elif c0 <= -3:
                                state[col_idx[j]] = -1
                                print("{0} Turn OFF!!!".format(id))
                            else:
                                state[col_idx[j]] = 0
                            INFO[ID[j]] = [1,Loc[col_idx[j]],bright,c,Cls[col_idx[j]]]
                            match[col_idx[j]] = ID[j]
                            # print("find ",col_idx[j],ID[j])
                        elif len(LOC) < len(Loc): # new device
                            col = int(-0.1-col_idx[j])
                            m = False
                            for k,Info in enumerate(INFO):
                                if Info[0] == 1:
                                    continue
                                else:
                                    if Match_loc(0.4,Loc[col],Info[1]) & int(Cls[col]) == int(Info[4]):
                                        INFO[k] = [1,Loc[col],B[col],0,Cls[col]]
                                        match[int(col)] = k
                                        m = True
                                        # print("  Match Loc:",INFO[k])
                                        break
                            if ~m:
                                INFO.append([1,Loc[col],B[col],0,Cls[col]])
                                # print(INFO)
                                match[int(col)] = len(INFO) # new id
                                # print("  new device!",Cls[col])
                        else:
                            INFO[ID[j]][0] = 0 # untracked device
                    # print("  INFO:",INFO)
                    # print("  ID:",ID)
                    # print("  match: ",match)
                else:
                    match = []
                    state = np.zeros(len(Loc))
                    # print("Loc: ",Loc)
                    for j in range(len(Loc)):
                        match.append(j)
                        loc = Loc[j]
                        img_temp = imc[loc[1]:loc[3],loc[0]:loc[2]] 
                        INFO.append([1,Loc[j],Brightness(img_temp),0,Cls[j]])
                        # print(INFO)
                for idx, info in enumerate(INFO):
                    if (info[0] != 0) & (int(info[4]) == 4): # exsit person
                        info.append(idx)
                        People.append(info)
                    if (info[0] != 0) & (int(info[4]) == 1): # exsit Phone
                        loc = info[1]
                        loc_w = int((loc[0]+loc[2])/2)
                        loc_h = int((loc[1]+loc[3])/2)
                        if (loc_w>100) & (loc_w<540) & (loc_h>320) & (loc_h<450):
                            situation0[1] = idx
                if (situation0[1] != -1) & (user_id != -1):
                    situation0[0] = user_id
                    print("situation 0:",situation0[0],situation0[1])
                else:
                    situation0 = [-1,-1]

                situation2_0 = situation2[0] 
                situation2[0] = -1 # ????
                for person in People:
                    if sit_2(w,h,person[1]):
                        if situation2[0] != -1:
                            situation2[0] = -1 # more than one person, what if someone sit behind you?? 
                            break
                        situation2[0] = person[5]
                        # print(person,"__",situation2[0])
                    else:
                        situation2[0] = -1
                if (situation2[0] == situation2_0) & (situation2[0] != -1): # same as last frame
                    if situation2[1] < 3:
                        situation2[1] += 1
                    elif situation2[1] == 3:
                        situation2[1] += 1
                        print("situation 2:",situation2,person)
                else: # different from last frame or no person
                    situation2[1] = 0

                
                temp = 0
                for *xyxy, conf, cls in reversed(det):
                    id = match[temp]
                    if state[temp] == 1:
                        S = "ON"
                    elif state[temp] == -1:
                        S = "OFF"
                    else:
                        S = ""
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # w, h = im0.shape[1], im0.shape[0]
                    if save_txt:  # Write to file
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{str(int(id))+" "+names[c]+" "+S} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    temp += 1
            
            # face recognition
            minW,minH = imFR.shape[1]*0.1,imFR.shape[0]*0.1
            gray = cv2.cvtColor(imFR,cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor = 1.1,
                minNeighbors = 3,
                minSize = (int(minW),int(minH))
                )
            for(x,y,w,h) in faces:
                cv2.rectangle(im0,(x,y),(x+w,y+h),(0,255,0),2)
                idnum,confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
                #计算出一个检验结果
                if confidence < 110:
                    idum = names_FR[idnum]
                    for person in People:
                        x0,y0,x1,y1 = person[1]
                        if (x>x0) & (y>y0) & (x+w<x1) & (y+h<y1):
                            user_id = person[5]
                            print("user_id",user_id)
                else:
                    idum = "unknown"
                #输出检验结果以及用户名
                cv2.putText(im0,str(idum),(x+5,y-5),font,1,(0,0,255),1)
                cv2.putText(im0,str(confidence),(x+5,y+h-5),font,1,(0,0,0),1)
            if user_id != -1:
                if INFO[user_id][0] == 0: # user disappear
                    user_id = -1
                    cv2.putText(im0,"user disappear",(10,50),font,1,(255,255,255),1)
                    # print("user disappear")
                else:
                    cv2.putText(im0,"user id:"+str(user_id),(10,50),font,1,(255,255,255),1)
                    user_loc = INFO[user_id][1]
                    
            t_mp = time_sync()
            Hands = []
            # print(imMP.shape)
            # imMP = np.transpose(imMP,(1,2,0))
            
            #print(imMP.shape)
            imMP.flags.writeable = False
            imMP = cv2.cvtColor(imMP, cv2.COLOR_BGR2RGB) 
            results_hands = hands.process(imMP)
            results_face_detection = face_detection.process(imMP)
            results_face_mesh = face_mesh.process(imMP)
            # if user_id != -1:
            #     results_face_mesh = face_mesh.process(imMP[loc[1]:loc[3],loc[0]:loc[2]])
            # else:
            #     results_face_mesh = face_mesh.process(imMP)

            # Draw the hand annotations on the image.
            imMP.flags.writeable = True
            # imMP = cv2.cvtColor(imMP, cv2.COLOR_RGB2BGR) 
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    x0,y0,x1,y1 = 1,1,0,0
                    for c in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[c].x
                        y = hand_landmarks.landmark[c].y
                        c += 1
                        if (x<x0) & (x>0):
                            x0 = x
                        if (x>x1) & (x<1):
                            x1 = x
                        if (y<y0) & (y>0):
                            y0 = y
                        if (y>y1) & (y<1):
                            y1 = y
                    # mp_drawing.draw_landmarks(
                    #     imMP,
                    #     hand_landmarks,
                    #     mp_hands.HAND_CONNECTIONS,
                    #     mp_drawing_styles.get_default_hand_landmarks_style(),
                    #     mp_drawing_styles.get_default_hand_connections_style())
                    Hands.append([x0,y0,x1,y1])
            else:
                0
                # print("No hands!")
                # cv2.imwrite(os.path.join(save_dir,str(frames) +".jpg"),imMP)
            if results_face_mesh.multi_face_landmarks:
                for face_landmarks in results_face_mesh.multi_face_landmarks:
                    # for c in range(len(face_landmarks.landmark)):
                    #     x = face_landmarks.landmark[c].x
                    #     y = face_landmarks.landmark[c].y
                    #     cv2.circle(image, (int(x*w),int(y*h)), 1, (255,0,0), 4)
                    #     print("x,y = ",int(x*w),int(y*h))
                    #     c += 1
                    # mp_drawing.draw_landmarks(
                    #     image=imMP,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACEMESH_TESSELATION,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=mp_drawing_styles
                    #     .get_default_face_mesh_tesselation_style())
                    # mp_drawing.draw_landmarks(
                    #     image=imMP,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACEMESH_CONTOURS,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=mp_drawing_styles
                    #     .get_default_face_mesh_contours_style())
                    # mp_drawing.draw_landmarks(
                    #     image=imMP,
                    #     landmark_list=face_landmarks,
                    #     connections=mp_face_mesh.FACEMESH_IRISES,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=mp_drawing_styles
                    #     .get_default_face_mesh_iris_connections_style())
                    1
                X1,Y1,Z1 = face_landmarks.landmark[mark[0]].x,face_landmarks.landmark[mark[0]].y,face_landmarks.landmark[mark[0]].z
                X2,Y2,Z2 = face_landmarks.landmark[mark[1]].x,face_landmarks.landmark[mark[1]].y,face_landmarks.landmark[mark[1]].z
                X3,Y3,Z3 = face_landmarks.landmark[mark[2]].x,face_landmarks.landmark[mark[2]].y,face_landmarks.landmark[mark[2]].z
                X,Y,Z = face_landmarks.landmark[mark[3]].x,face_landmarks.landmark[mark[3]].y,face_landmarks.landmark[mark[3]].z

                a = Y2*Z3 - Y2*Z1 - Y1*Z3 - Y3*Z2 + Y1*Z2 + Y3*Z1
                b = X3*Z2 - X1*Z2 - X3*Z1 - X2*Z3 + X2*Z1 + X1*Z3
                c = X2*Y3 - X2*Y1 - X1*Y3 - X3*Y2 + X3*Y1 + X1*Y2

                direction1 = "left" if a<-0.0015 else ("right" if a>0.0015 else "middle")
                direction2 = "up" if b>-0.002 else "down"
                # print("%s, %s: a=%.4f, b=%.4f, c=%.4f"%(direction1,direction2,a,b,c))
            
            if results_face_detection.detections:
                if len(results_face_detection.detections) > 1:
                    if count_sit3 > 3:
                        situation3 = True
                    else:
                        count_sit3 += 1
            else:
                count_sit3 = 0
                situation3 = False
            print("situation 3:",situation3)
                

            t_MP += (time_sync()-t_mp)
            # print("MediaPipe hands: %.4fs" % (time_sync()-t_mp))
            frames += 1
            
            # Stream results
            im0 = annotator.result()
            # a = w/im0.shape[0]
            w,h = im0.shape[1],im0.shape[0]
            for hand in Hands:
                flag = -1
                x0,y0,x1,y1 = int(hand[0]*w),int(hand[1]*h),int(hand[2]*w),int(hand[3]*h)
                for idx,info in enumerate(INFO):
                    if (info[0] == 0) | (int(info[4]) == 4): # untracked device or person
                        continue
                    loc = info[1]
                    # print("loc:",loc)
                    minx = max(loc[0]*0.9+loc[2]*0.1, x0)
                    miny = max(loc[1]*0.9+loc[3]*0.1, y0)
                    maxx = min(loc[0]*0.1+loc[2]*0.9, x1)
                    maxy = min(loc[1]*0.1+loc[3]*0.9, y1)
                    if (minx < maxx) & (miny < maxy):
                    #if (((x0>loc[0])&(x0<loc[2])) | ((x1>loc[0])&(x1<loc[2]))) & (((y0>loc[1])&(y0<loc[3])) | ((y1>loc[1])&(y1<loc[3]))):
                        flag = idx # only record one device, maybe more
                        # print("  Match! Loc:", idx, info[4], loc)
                        # print("       hands:         ",x0,y0,x1,y1)
                        break

                if flag != -1:
                    cv2.rectangle(im0, (int(hand[0]*w),int(hand[1]*h)),(int(hand[2]*w),int(hand[3]*h)), (255,0,0), 2)
                else:
                    cv2.rectangle(im0, (int(hand[0]*w),int(hand[1]*h)),(int(hand[2]*w),int(hand[3]*h)), (0,255,0), 2)
                # print(w,h,int(hand[0]*w),int(hand[1]*h),int(hand[2]*w),int(hand[3]*h))
            
            if len_INFO != len(INFO):
                cv2.imwrite(os.path.join(save_dir,str(c_INFO)+".jpg"),im0s[i])
                len_INFO = len(INFO)
                c_INFO += 1
                print(os.path.join(save_dir,str(c_INFO)+".jpg"))

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        t4 = time_sync()
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s) ({t4 - t1:.3f}s)')
        T += (t4-t1)

    # Print results
    print(INFO)
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    LOGGER.info(f'Speed: %.1fms MediaPipe Hands per image at shape {(1, 3, 1080, 1920)}' % (t_MP*1E3/frames))
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp15/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    # 'datasets/test/WIN_20220420_11_46_51_Pro.mp4'
    parser.add_argument('--data', type=str, default=ROOT / 'data/MultiDevices.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    source = 'datasets/test/WIN_20220322_16_51_01_Pro.mp4'
    opt = parse_opt()
    main(opt)
