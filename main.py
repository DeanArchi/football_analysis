from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFileDialog,
    QLabel,
    QFrame,
    QListWidgetItem
)
from PyQt5.QtCore import QThread, pyqtSignal, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
import sys
import os
from utils import read_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistanceEstimator


class VideoProcessThread(QThread):
    finished = pyqtSignal(str, dict)
    progress = pyqtSignal(int)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            video_frames = read_video(self.video_path)

            # Initialize Tracker
            tracker = Tracker('models/best.pt')
            tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

            # Get object positions
            tracker.add_position_to_tracks(tracks)

            # Camera movement estimator
            camera_movement_estimator = CameraMovementEstimator(video_frames[0])
            camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
                video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stub.pkl'
            )
            camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

            # View Transformer
            view_transformer = ViewTransformer()
            view_transformer.add_transformed_position_to_tracks(tracks)

            # Interpolate Ball Positions
            tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

            # Speed and distance estimator
            speed_and_distance_estimator = SpeedAndDistanceEstimator()
            speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

            # Assign Player Teams
            team_assigner = TeamAssigner()
            team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
            for frame_num, player_track in enumerate(tracks['players']):
                for player_id, track in player_track.items():
                    team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
                    tracks['players'][frame_num][player_id]['team'] = team
                    tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

            # Assign Ball Acquisition
            player_assigner = PlayerBallAssigner()
            team_ball_control = []
            for frame_num, player_track in enumerate(tracks['players']):
                ball_bbox = tracks['ball'][frame_num][1]['bbox']
                assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
                if assigned_player != -1:
                    tracks['players'][frame_num][assigned_player]['has_ball'] = True
                    team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
                else:
                    if len(team_ball_control) > 0:
                        team_ball_control.append(team_ball_control[-1])
                    else:
                        team_ball_control.append(0)

            team_ball_control = np.array(team_ball_control)

            # Draw output
            output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
            speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

            stats = self.calculate_statistics(tracks, team_ball_control)

            filename = os.path.basename(self.video_path)
            base_name, ext = os.path.splitext(filename)
            output_path = os.path.join('output_videos', f'processed_{base_name}.mp4')
            os.makedirs('output_videos', exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width = output_video_frames[0].shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

            for frame in output_video_frames:
                out.write(frame)
            out.release()

            self.finished.emit(output_path, stats)

        except Exception as e:
            print(f'Помилка обробки відео: {e}')
            self.finished.emit('', {})

    @staticmethod
    def calculate_statistics(tracks, team_ball_control):
        stats = {
            'team_possession': {
                'team1': (team_ball_control == 1).sum() / len(team_ball_control) * 100,
                'team2': (team_ball_control == 2).sum() / len(team_ball_control) * 100
            },
            'players': {}
        }

        for player_id in tracks['players'][-2].keys():
            player_speeds = []
            for frame in tracks['players']:
                if player_id in frame and 'speed' in frame[player_id]:
                    player_speeds.append(frame[player_id]['speed'])

            if player_speeds:
                avg_speed = sum(player_speeds) / len(player_speeds)
            else:
                avg_speed = 0

            last_frame = tracks['players'][-2]
            if player_id in last_frame:
                team = last_frame[player_id]['team']
                team_color = last_frame[player_id]['team_color'] if 'team_color' in last_frame[player_id] else (0, 0, 0)
                total_distance = last_frame[player_id].get('distance', 0)

                stats['players'][player_id] = {
                    'avg_speed': avg_speed,
                    'total_distance': total_distance,
                    'team': team,
                    'team_color': team_color
                }

        return stats


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Football Analysis')
        self.setGeometry(500, 250, 1200, 700)

        self.processed_videos = {}
        self.current_video = None

        self.upload_button = None
        self.video_list = None
        self.video_container = None
        self.media_player = None
        self.video_widget = None
        self.play_button = None
        self.pause_button = None
        self.player_status = None
        self.progress_label = None
        self.stats_container = None
        self.team1_layout = None
        self.team1_label = None
        self.team1_color = None
        self.team1_possession = None
        self.team2_layout = None
        self.team2_label = None
        self.team2_color = None
        self.team2_possession = None
        self.players_stats_container = None
        self.processing_thread = None

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()

        left_panel = QVBoxLayout()

        self.upload_button = QPushButton('Завантажити відео')
        self.upload_button.clicked.connect(self.upload_video)
        left_panel.addWidget(self.upload_button)

        self.video_list = QListWidget()
        self.video_list.currentItemChanged.connect(self.video_selected)
        left_panel.addWidget(self.video_list)

        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setFixedWidth(300)
        main_layout.addWidget(left_widget)

        right_panel = QVBoxLayout()

        self.video_container = QWidget()
        video_layout = QVBoxLayout(self.video_container)

        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.error.connect(self.handle_player_error)
        self.media_player.stateChanged.connect(self.media_state_changed)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(360)
        self.media_player.setVideoOutput(self.video_widget)
        video_layout.addWidget(self.video_widget)

        player_controls = QHBoxLayout()

        self.play_button = QPushButton('Відтворити')
        self.play_button.clicked.connect(self.play_video)
        player_controls.addWidget(self.play_button)

        self.pause_button = QPushButton('Пауза')
        self.pause_button.clicked.connect(self.pause_video)
        player_controls.addWidget(self.pause_button)

        video_layout.addLayout(player_controls)

        self.player_status = QLabel('Відео не завантажено')
        video_layout.addWidget(self.player_status)

        progress_container = QVBoxLayout()

        self.progress_label = QLabel('Відео обробляється, будь ласка зачекайте...')
        progress_container.addWidget(self.progress_label)

        self.progress_label.hide()

        video_layout.addLayout(progress_container)

        stats_widget = QWidget()
        self.stats_container = QVBoxLayout(stats_widget)

        stats_title = QLabel('Статистика матчу')
        stats_title.setStyleSheet('font-size: 18px; font-weight: bold;')
        self.stats_container.addWidget(stats_title)

        possession_layout = QVBoxLayout()
        possession_title = QLabel('Володіння м\'ячем:')
        possession_layout.addWidget(possession_title)

        self.team1_layout, self.team1_color, self.team1_possession = self.create_team_layout('Команда 1:')
        possession_layout.addLayout(self.team1_layout)

        self.team2_layout, self.team2_color, self.team2_possession = self.create_team_layout('Команда 2:')
        possession_layout.addLayout(self.team2_layout)

        self.stats_container.addLayout(possession_layout)

        players_title = QLabel('Статистика гравців:')
        players_title.setStyleSheet('font-size: 16px; font-weight: bold; margin-top: 10px;')
        self.stats_container.addWidget(players_title)

        self.players_stats_container = QVBoxLayout()
        self.stats_container.addLayout(self.players_stats_container)

        right_panel.addWidget(self.video_container, 6)
        right_panel.addWidget(stats_widget, 4)

        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        main_layout.addWidget(right_widget)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    @staticmethod
    def create_team_layout(team_name):
        layout = QHBoxLayout()
        label = QLabel(team_name)
        color_frame = QFrame()
        color_frame.setFixedSize(20, 20)
        possession_label = QLabel('0%')
        layout.addWidget(label)
        layout.addWidget(color_frame)
        layout.addWidget(possession_label)
        layout.addStretch()
        return layout, color_frame, possession_label

    def upload_video(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, 'Завантажити відео', '', 'Video Files (*.mp4 *.avi *.mov *.mkv)'
        )

        if file_path:
            self.progress_label.show()

            self.processing_thread = VideoProcessThread(file_path)
            self.processing_thread.finished.connect(self.video_processed)
            self.processing_thread.start()

    def video_processed(self, output_path, stats):
        self.progress_label.hide()

        if output_path:
            video_name = os.path.basename(output_path)

            if not os.path.exists(output_path):
                print(f'ПОМИЛКА: Файл {output_path} не знайдено!')
                return

            self.processed_videos[video_name] = {
                'path': output_path,
                'stats': stats
            }

            item = QListWidgetItem(video_name)
            self.video_list.addItem(item)

            self.video_list.setCurrentItem(item)

            print(f'Відео оброблено і збережено в: {output_path}')

    def video_selected(self, current):
        if current:
            video_name = current.text()
            if video_name in self.processed_videos:
                video_data = self.processed_videos[video_name]
                self.current_video = video_data['path']

                if not os.path.exists(self.current_video):
                    self.player_status.setText(f'ПОМИЛКА: Файл {self.current_video} не знайдено!')
                    return

                print(f'Завантаження відео: {self.current_video}')

                self.load_video(self.current_video)

                self.display_statistics(video_data['stats'])

    def load_video(self, video_path):
        abs_path = os.path.abspath(video_path)
        print(f'Абсолютний шлях до відео: {abs_path}')

        if not os.path.exists(abs_path):
            self.player_status.setText(f'ПОМИЛКА: Файл не знайдено: {abs_path}')
            return

        url = QUrl.fromLocalFile(abs_path)
        print(f'URL для відтворення: {url.toString()}')

        content = QMediaContent(url)
        self.media_player.setMedia(content)

        self.media_player.play()
        self.video_widget.show()

    def play_video(self):
        if self.media_player.mediaStatus() == QMediaPlayer.NoMedia:
            self.player_status.setText('Немає завантаженого відео')
            return

        self.media_player.play()
        self.player_status.setText('Відтворення...')

    def pause_video(self):
        self.media_player.pause()

    def handle_player_error(self, error):
        error_messages = {
            QMediaPlayer.NoError: 'Немає помилок',
            QMediaPlayer.ResourceError: 'Ресурс не знайдено або недоступний',
            QMediaPlayer.FormatError: 'Формат не підтримується',
            QMediaPlayer.NetworkError: 'Помилка мережі',
            QMediaPlayer.AccessDeniedError: 'Доступ заборонено',
            QMediaPlayer.ServiceMissingError: 'Сервіс не знайдено'
        }

        error_text = error_messages.get(error, f'Невідома помилка: {error}')
        self.player_status.setText(f'Помилка відтворення: {error_text}')
        print(f'Помилка плеєра: {error_text}')

    def media_state_changed(self, state):
        states = {
            QMediaPlayer.StoppedState: 'Зупинено',
            QMediaPlayer.PlayingState: 'Відтворення',
            QMediaPlayer.PausedState: 'Пауза'
        }

        self.player_status.setText(states.get(state, f'Невідомий стан: {state}'))

    def display_statistics(self, stats):
        self.clear_layout(self.players_stats_container)

        if 'team_possession' in stats:
            team1_possession = stats['team_possession'].get('team1', 0)
            team2_possession = stats['team_possession'].get('team2', 0)

            self.team1_possession.setText(f'{team1_possession:.1f}%')
            self.team2_possession.setText(f'{team2_possession:.1f}%')

            if 'players' in stats and stats['players']:
                team_colors = {}
                for player_id, player_stats in stats['players'].items():
                    team = player_stats.get('team')
                    if team and 'team_color' in player_stats:
                        team_colors[team] = player_stats['team_color']

                if 1 in team_colors:
                    color = team_colors[1]
                    self.team1_color.setStyleSheet(f'background-color: rgb({color[0]}, {color[1]}, {color[2]});')

                if 2 in team_colors:
                    color = team_colors[2]
                    self.team2_color.setStyleSheet(f'background-color: rgb({color[0]}, {color[1]}, {color[2]});')

        if 'players' in stats:
            for player_id, player_stats in stats['players'].items():
                player_layout = QHBoxLayout()

                team_color_frame = QFrame()
                team_color_frame.setFixedSize(15, 15)
                if 'team_color' in player_stats:
                    color = player_stats['team_color']
                    team_color_frame.setStyleSheet(f'background-color: rgb({color[0]}, {color[1]}, {color[2]});')
                player_layout.addWidget(team_color_frame)

                player_id_label = QLabel(f'Гравець {player_id}:')
                player_layout.addWidget(player_id_label)

                avg_speed = player_stats.get('avg_speed', 0)
                speed_label = QLabel(f'Швидкість: {avg_speed:.2f} км/год')
                player_layout.addWidget(speed_label)

                total_distance = player_stats.get('total_distance', 0)
                distance_label = QLabel(f'Дистанція: {total_distance:.2f} м')
                player_layout.addWidget(distance_label)

                player_layout.setSpacing(3)

                player_layout.addStretch()
                self.players_stats_container.addLayout(player_layout)

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())


def application():
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    application()
