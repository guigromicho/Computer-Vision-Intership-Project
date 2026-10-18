
#!pip install opencv-python
#!pip install ultralytics
#!pip install pyyaml

import os
import ultralytics
import cv2
import numpy as np
import time

#class to save video information
class Video:
    def __init__(self, video_path):

        #video info
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 10
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


        #big mask (tactile line and train track)
        self.tactile_masks = None
        self.tactile_detected = False

        self.train_track_masks = None
        self.train_track_detected = False
        self.frames_between_detections = 100


        #file to export after rendering
        output_name = f"output_{os.path.basename(video_path)}"
        self.out = cv2.VideoWriter(
            output_name,
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps,
            (self.width, self.height)
        )

        #dic for diferent colors depending on name detection
        self.colors = {
            "tactile": (255, 50, 50),
            "person": (100, 255, 100),
            "train track": (150, 150, 255),
            "head lights": (200, 0, 200),
            "train": (250, 0, 250)
        }

    #detect if a point is inside other mask
    def is_on_mask(self, mask, cx, cy, check_height=30, check_width=30):
        if mask is None:
            return False

        x1 = max(cx - check_width // 2, 0)
        x2 = min(cx + check_width // 2, self.width)

        y1 = cy
        y2 = min(cy + check_height, self.height)

        region = mask[y1:y2, x1:x2]

        return np.any(region == 1)
    
    #detect if person poins are inside of train track or tactile line  
    def is_person_on(self, points):
        for (cx, cy) in points:
            if self.is_on_mask(self.train_track_masks, cx, cy):
                return True, (cx, cy)

            if self.is_on_mask(self.tactile_masks, cx, cy):
                return True, (cx, cy)

        return False, None
    #draw bounding boxes
    def draw_box(self, frame, results):
        if results.boxes is None:
            return frame

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls)
            conf = float(box.conf)

            color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame,f"{results.names[cls]} {conf:.2f}",(x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2,)

        return frame

    #draw segmentation mask
    def draw_seg(self, frame, results):
        if results.masks is None:
            return frame

        annotated = frame.copy()
        masks = results.masks.data.cpu().numpy()

        for i, mask in enumerate(masks):
            cls_id = int(results.boxes[i].cls)
            class_name = results.names[cls_id]

            if class_name == "person":
                continue
            if class_name == "head lights":
              continue

            mask = cv2.resize(mask.astype(np.uint8), (self.width, self.height))
            mask = (mask > 0.5).astype(np.uint8)

            colored = np.zeros_like(frame)

            if class_name in self.colors:
                b, g, r = self.colors[class_name]
                colored[mask == 1] = (b, g, r)
            else:
                colored[mask == 1] = (0, 255, 0)

            annotated = cv2.addWeighted(annotated, 1, colored, 0.4, 0)

        return annotated

    #get object mask
    def get_mask(self,results, class_name):
        if results.masks is None:
            return None

        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes

        for i, mask in enumerate(masks):
            cls_id = int(boxes[i].cls)
            cls_name = results.names[cls_id]

            if cls_name == class_name:
                mask = cv2.resize(mask.astype(np.uint8), (self.width, self.height))
                return (mask > 0.5).astype(np.uint8)

        return None

    #compare area of masks
    def compare_masks(self, mask1, mask2):
        if mask1 is None:
            return mask2
        if mask2 is None:
            return mask1

        area1 = np.sum(mask1)
        area2 = np.sum(mask2)

        return mask1 if area1 > area2 else mask2

    #get the 3 bottom points of person bouding box
    def get_person_bottom_points(self, results):
        points = []

        if results.boxes is None:
            return points

        for box in results.boxes:
            cls = int(box.cls)

            if results.names[cls] != "person":
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            cx = int((x1 + x2) / 2)
            cy = int(y2)

            points.append((cx, cy))
            points.append((x2, y2))
            points.append((x1, y2))

        return points

    #draw person points
    def draw_person_points(self, frame, points):
        for (cx, cy) in points:
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
        return frame

    def detect():
        pass


#load the video from the folder
def load_video(video_path, model, skip_frames,verbose):
    print(f"Loading video: {video_path}")
    video = Video(video_path)

    frame_count = 0
    start = time.time()

    while True:
        ret, frame = video.cap.read()
        if not ret:
            break
        # detect classes from drame
        if frame_count % skip_frames == 0:

            classes = [0, 1, 3]
            
            #append mask if the havent been detected
            if not video.tactile_detected:
                classes.append(2)

            if not video.train_track_detected:
                classes.append(4)

            results = model(frame, classes=classes, verbose=verbose)[0]

            if not video.tactile_detected:
                video.get_mask(results, "tactile")
                video.tactile_detected = True

            if not video.train_track_detected:
                video.get_mask(results, "train track")
                video.train_track_detected = True

            frame = video.draw_box(frame, results)
            frame = video.draw_seg(frame, results)

            points = video.get_person_bottom_points(results)
            frame = video.draw_person_points(frame, points)

            on_track, hit_point = video.is_person_on(points)
            print(on_track)

            #draw if person is in danger
            if on_track:
                cx, cy = hit_point
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
                cv2.putText(frame, "Person in danger",(cx + 10, cy - 10),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)

            if (frame_count + 1) % video.frames_between_detections == 0:
                video.tactile_detected = False
                video.train_track_detected = False

        video.out.write(frame)
        frame_count += 1

    video.cap.release()
    video.out.release()

    #time to render the video
    end = time.time()
    print(f"Tempo: {end - start:.2f}s")

#load model from file
def load_model(model_path):
    print(f"Loading model: {model_path}")
    return ultralytics.YOLO(model_path)

#main funtion
def main():
    model = load_model("/content/exp-2.pt")
    load_video("/content/Man falls off Boston subway platform onto train tracks.mp4",model,skip_frames=1,verbose=False)


if __name__ == "__main__":
    main()
    