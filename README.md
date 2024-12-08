# Face Emotion Detection

This is the second part of my educational project on face emotion detection.  
The project combines face detection and emotion classification in real-time using a camera feed.

### Project Overview:
1. Emotion Recognition Model: I first developed an [Emotion Recognition CNN](https://github.com/Danchery/Emotion-recognition/tree/master).
2. Face Detection: Fine-tuned YOLOv8n from Ultralytics on the [Open Images Dataset V7](https://storage.googleapis.com/openimages/web/index.html) to detect faces.

---

## Results
![test_2](https://github.com/user-attachments/assets/3366188a-414d-494f-8a3f-2e833bc7bcfb)


---

## Project Structure
 
- `models/`: Contains the fine-tuned weights for the detection and classification models.
- `notebooks/`: Includes the pipeline for combining detection and classification, along with helper functions.
- `Results/` : Demonstration Gif

---

## Future Work
- Improve robustness of the system for emotion recognition
- Experiment with more datasest and architectures for build more robust emotion recognitional system.

---

## Tools and Libraries
- PyTorch: For training and fine-tuning models.
- Ultralytics: For face detection.
- OpenCV: For handling video feeds and image processing.

