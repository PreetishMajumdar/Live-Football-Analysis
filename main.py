from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import threading
import queue
import time

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(1)
        self.stopped = False
        self.Q = queue.Queue(maxsize=128)
        self.frame_count = 0

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            ret, frame = self.stream.read()
            if not ret:
                self.stop()
                return
            if self.Q.full():
                self.Q.get_nowait()
            self.Q.put(frame)
            self.frame_count += 1

    def read(self):
        return self.Q.get()

    def stop(self):
        self.stopped = True

class RealTimeProcessor:
    def __init__(self):
        self.tracker = Tracker('models/best.pt')
        self.camera_movement_estimator = None
        self.view_transformer = ViewTransformer()
        self.speed_estimator = SpeedAndDistance_Estimator()
        self.team_assigner = TeamAssigner()
        self.player_assigner = PlayerBallAssigner()
        self.team_ball_control = []
        self.prev_positions = {}
        self.lock = threading.Lock()

    def initialize_components(self, first_frame):
        self.camera_movement_estimator = CameraMovementEstimator(first_frame)
        self.team_assigner.assign_team_color(first_frame, {})

    def process_frame(self, frame):
        with self.lock:
            # Object detection and tracking
            tracks = self.tracker.get_object_tracks([frame], read_from_stub=False)
            self.tracker.add_position_to_tracks(tracks)
            
            # Camera movement compensation
            camera_movement = self.camera_movement_estimator.get_camera_movement([frame])
            self.camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement)
            
            # View transformation
            self.view_transformer.add_transformed_position_to_tracks(tracks)
            
            # Ball interpolation
            tracks["ball"] = self.tracker.interpolate_ball_positions(tracks["ball"])
            
            # Speed and distance calculation
            self.speed_estimator.add_speed_and_distance_to_tracks(tracks)
            
            # Team assignment
            for player_id, track in tracks['players'][0].items():
                team = self.team_assigner.get_player_team(frame, track['bbox'], player_id)
                tracks['players'][0][player_id]['team'] = team
                tracks['players'][0][player_id]['team_color'] = self.team_assigner.team_colors[team]
            
            # Ball possession
            ball_bbox = tracks['ball'][0][1]['bbox'] if tracks['ball'] else None
            if ball_bbox:
                assigned_player = self.player_assigner.assign_ball_to_player(tracks['players'][0], ball_bbox)
                if assigned_player != -1:
                    self.team_ball_control.append(tracks['players'][0][assigned_player]['team'])
                else:
                    self.team_ball_control.append(self.team_ball_control[-1] if self.team_ball_control else 0)
            
            # Draw annotations
            annotated_frame = self.tracker.draw_annotations([frame], tracks, np.array(self.team_ball_control))[0]
            annotated_frame = self.speed_estimator.draw_speed_and_distance([annotated_frame], tracks)[0]
            
            return annotated_frame

def main():
    # Initialize video stream (0 for webcam or RTSP URL)
    video_stream = VideoStream(0).start()
    time.sleep(1.0)  # Allow camera to warm up

    # Get first frame for initialization
    first_frame = video_stream.read()
    
    # Initialize processing pipeline
    processor = RealTimeProcessor()
    processor.initialize_components(first_frame)
    
    # Create display thread
    display_queue = queue.Queue(maxsize=10)
    
    def display_worker():
        while True:
            frame = display_queue.get()
            if frame is None:
                break
            cv2.imshow("Live Football Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    display_thread = threading.Thread(target=display_worker, daemon=True)
    display_thread.start()

    # Main processing loop
    try:
        while True:
            frame = video_stream.read()
            processed_frame = processor.process_frame(frame)
            
            if display_queue.full():
                display_queue.get_nowait()
            display_queue.put(processed_frame)
            
    except KeyboardInterrupt:
        pass

    # Cleanup
    video_stream.stop()
    display_queue.put(None)
    display_thread.join()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
