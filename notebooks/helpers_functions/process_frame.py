
import torch
import torchvision
import cv2
from ultralytics import YOLO
from PIL import Image

def process_frame(frame, detector, classifier_transforms, classifier, classes, device):
    """
    Procces the given frame input with detection and classification models
    """
    #detection
    results = detector(frame, verbose=False)
    
    detections = results[0]

    for detection in detections:
    
        x_center, y_center, width, height  = detection.boxes.xywh.tolist()[0]
        start_y = float(y_center - height / 2)
        start_x = float(x_center - width / 2)
        width = float(width)
        height = float(height)
        
        picture = Image.fromarray(results[0].orig_img)
        croped_picture = torchvision.transforms.functional.crop(picture, start_y, start_x, height, width)
        
        #plt.imshow(croped_picture)
        
        transformed_picture = classifier_transforms(croped_picture).unsqueeze(0).to(device)
        #plt.imshow(torchvision.transforms.functional.to_pil_image(transformed_picture.squeeze(0)))
        
        #classification
        classifier.eval()
        with torch.inference_mode():
            logits = classifier(transformed_picture.to(device))
            pred_prob = torch.softmax(logits, dim=1)
            proba = max(pred_prob.tolist()[0])
            emotion = classes[torch.argmax(pred_prob, dim=1).item()]
    
        start_x, start_y, end_x, end_y = detection.boxes.xyxy.tolist()[0]
        start_x = int(start_x)
        start_y = int(start_y)
        end_x = int(end_x)
        end_y = int(end_y)
    
        frame = results[0].orig_img

        #draw bb
        color = (0, 191, 255)
        thickness = 2
        cv2.rectangle(img=frame, pt1=(start_x, start_y), pt2=(end_x, end_y), color=color, thickness = thickness)

        # put the text on the image
        org = (start_x, start_y)# - 20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        text = f'{emotion} - {round(proba, 3)}'
        cv2.putText(frame, text, org, font, fontScale, color, thickness)
        #frame = Image.fromarray(frame[..., ::-1])# for image

    return frame
