from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

results = model('sample_video.MOV', show=True, save=True, save_txt=True, conf=0.3)