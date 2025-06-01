from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from utility.bbox_utils import BBoxUtils

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self, tracks):
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object_type == 'ball':
                        position = BBoxUtils.get_center_of_bbox(bbox)
                    else:
                        position = BBoxUtils.get_foot_position(bbox)
                    track_info['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        if not ball_positions:
            return []

        try:
            if not all(len(pos) == 4 for pos in ball_positions):
                raise ValueError("Invalid ball position format")

            df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

            df_ball_positions = df_ball_positions.interpolate().bfill()
            return df_ball_positions.values.tolist()

        except Exception as e:
            print(f"Ball interpolation error: {str(e)}")
            return []

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            for i, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[i] = cls_names_inv["player"]

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = BBoxUtils.get_center_of_bbox(bbox)
        width = BBoxUtils.get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        if track_id is not None:
            rect_w, rect_h = 40, 20
            x1 = x_center - rect_w // 2
            x2 = x_center + rect_w // 2
            y1 = y2 - rect_h // 2 + 15
            y2_ = y2 + rect_h // 2 + 15

            cv2.rectangle(frame, (x1, y1), (x2, y2_), color, cv2.FILLED)
            x_text = x1 + 12 - (10 if track_id > 99 else 0)

            cv2.putText(frame, f"{track_id}", (x_text, y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = BBoxUtils.get_center_of_bbox(bbox)
        triangle = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [triangle], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle], 0, (0, 0, 0), 2)
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        if len(team_ball_control) <= frame_num:
            print(f"Warning: Frame index {frame_num} out of range for team_ball_control.")
            return frame

        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        control_until_now = team_ball_control[:frame_num + 1]
        team_1 = np.count_nonzero(control_until_now == 1)
        team_2 = np.count_nonzero(control_until_now == 2)

        total = team_1 + team_2
        if total == 0:
            return frame

        team_1_pct = team_1 / total
        team_2_pct = team_2 / total

        cv2.putText(frame, f"Team 1 Ball Control: {team_1_pct * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2_pct * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_frames = []

        for frame_num, frame in enumerate(video_frames):
            if (frame_num >= len(tracks["players"]) or
                frame_num >= len(tracks["ball"]) or
                frame_num >= len(tracks["referees"])):
                print(f"Warning: Frame index {frame_num} out of range for tracks.")
                continue

            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)
                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            for referee in referee_dict.values():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            for ball in ball_dict.values():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            output_frames.append(frame)

        return output_frames
