from sklearn.cluster import KMeans
import numpy as np
import cv2

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None
        self.color_diff_threshold = 50  # Minimum BGR color difference between teams

    def get_clustering_model(self, image):
        if image.size == 0:
            return None
            
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        try:
            x1, y1, x2, y2 = map(int, bbox)
            image = frame[y1:y2, x1:x2]
            
            if image.size == 0:
                return np.array([0, 0, 0])
                
            top_half_image = image[0:int(image.shape[0]/2), :]
            kmeans = self.get_clustering_model(top_half_image)
            
            if not kmeans:
                return np.array([0, 0, 0])
                
            clustered_image = kmeans.labels_.reshape(top_half_image.shape[0], top_half_image.shape[1])
            corner_clusters = [
                clustered_image[0, 0], clustered_image[0, -1],
                clustered_image[-1, 0], clustered_image[-1, -1]
            ]
            non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
            player_cluster = 1 - non_player_cluster
            return kmeans.cluster_centers_[player_cluster]
            
        except Exception as e:
            return np.array([0, 0, 0])

    def assign_team_color(self, frame, player_detections):
        if len(player_detections) < 2:
            return  # Need at least 2 players to determine teams
            
        player_colors = []
        for _, detection in player_detections.items():
            color = self.get_player_color(frame, detection["bbox"])
            player_colors.append(color)
        
        player_colors = np.array(player_colors)
        
        try:
            # Check color variance before clustering
            if np.var(player_colors) < 100:
                return
                
            self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
            self.kmeans.fit(player_colors)
            
            # Ensure sufficient color difference between teams
            center_diff = np.linalg.norm(self.kmeans.cluster_centers_[0] - self.kmeans.cluster_centers_[1])
            if center_diff > self.color_diff_threshold:
                self.team_colors[1] = self.kmeans.cluster_centers_[0]
                self.team_colors[2] = self.kmeans.cluster_centers_[1]
                self.player_team_dict.clear()  # Reset cache when teams change
        except Exception as e:
            pass

    def get_player_team(self, frame, player_bbox, player_id):
        # Return cached value if available
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
            
        if not self.kmeans or len(self.team_colors) < 2:
            return 0  # Return 0 for unknown team
            
        try:
            player_color = self.get_player_color(frame, player_bbox)
            if np.linalg.norm(player_color) < 1:  # Invalid color detection
                return 0
                
            team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1
            
            # Verify against team colors
            if team_id not in self.team_colors:
                return 0
                
            self.player_team_dict[player_id] = team_id
            return team_id
            
        except Exception as e:
            return 0
