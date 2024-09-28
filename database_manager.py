# database_manager.py

import os
import numpy as np
import pickle

class DatabaseManager:
    def __init__(self, db_path='database'):
        self.db_path = db_path
        if not os.path.exists(db_path):
            os.makedirs(db_path)
        self.index_file = os.path.join(db_path, 'index.pkl')
        self.load_index()

    def load_index(self):
        if os.path.exists(self.index_file):
            with open(self.index_file, 'rb') as f:
                self.index = pickle.load(f)
        else:
            self.index = {}

    def save_index(self):
        with open(self.index_file, 'wb') as f:
            pickle.dump(self.index, f)

    def add_video(self, video_id, features):
        feature_file = os.path.join(self.db_path, f"{video_id}.npy")
        np.save(feature_file, features)
        self.index[video_id] = feature_file
        self.save_index()

    def get_all_features(self):
        features_list = []
        video_ids = []
        for video_id, feature_file in self.index.items():
            if os.path.exists(feature_file):
                features = np.load(feature_file)
                features_list.append(features)
                video_ids.append(video_id)
            else:
                print(f"Файл признаков не найден: {feature_file}")
        if features_list:
            return np.array(features_list), video_ids
        else:
            return np.array([]), []

    def delete_video(self, video_id):
        if video_id in self.index:
            feature_file = self.index[video_id]
            if os.path.exists(feature_file):
                os.remove(feature_file)
            del self.index[video_id]
            self.save_index()
            print(f"Видео {video_id} удалено из базы данных.")
        else:
            print(f"Видео {video_id} не найдено в базе данных.")
