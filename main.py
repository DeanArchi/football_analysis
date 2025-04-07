from PyQt5.QtGui import QIcon
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
    QListWidgetItem, QScrollArea
)
from PyQt5.QtCore import QThread, pyqtSignal, QUrl, Qt
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
            tracks = tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path='stubs/track_stubs.pkl')

            # Get object positions
            tracker.add_position_to_tracks(tracks)

            # Camera movement estimator
            camera_movement_estimator = CameraMovementEstimator(video_frames[0])
            camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
                video_frames, read_from_stub=False, stub_path='stubs/camera_movement_stub.pkl'
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
        total_frames = len(team_ball_control)
        if total_frames > 0:
            team1_percentage = (team_ball_control == 1).sum() / total_frames * 100
            team2_percentage = (team_ball_control == 2).sum() / total_frames * 100
        else:
            team1_percentage = 0
            team2_percentage = 0

        stats = {
            'team_possession': {
                'team1': team1_percentage,
                'team2': team2_percentage
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

        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f2f5;
            }
            QLabel {
                font-family: 'Segoe UI', Arial, sans-serif;
                color: #2c3e50;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1c6ea4;
            }
            QListWidget {
                background-color: white;
                border: 1px solid #dcdde1;
                border-radius: 4px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #f0f0f0;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f2f5;
                width: 8px;
                margin: 0px 0px 0px 0px;
            }
            QScrollBar::handle:vertical {
                background: #bdc3c7;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #95a5a6;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

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
        main_layout.setSpacing(15)  # Додаємо більше простору між панелями

        # Ліва панель (список відео)
        left_panel = QVBoxLayout()
        left_panel.setContentsMargins(10, 10, 10, 10)

        # Заголовок лівої панелі
        left_title = QLabel('Відеофайли')
        left_title.setStyleSheet('font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #34495e;')
        left_panel.addWidget(left_title)

        # Кнопка завантаження
        self.upload_button = QPushButton('Завантажити відео')
        self.upload_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-size: 14px;
                padding: 10px;
                margin-bottom: 15px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:pressed {
                background-color: #219653;
            }
        """)
        self.upload_button.setIcon(QIcon.fromTheme("document-open"))
        self.upload_button.clicked.connect(self.upload_video)
        left_panel.addWidget(self.upload_button)

        # Список відео
        list_container = QVBoxLayout()
        list_label = QLabel('Оброблені відео:')
        list_label.setStyleSheet('font-size: 14px; font-weight: bold; margin-bottom: 5px;')
        list_container.addWidget(list_label)

        self.video_list = QListWidget()
        self.video_list.setStyleSheet("""
            QListWidget {
                background-color: white;
                border-radius: 6px;
                padding: 5px;
            }
            QListWidget::item {
                height: 35px;
                padding-left: 10px;
                border-bottom: 1px solid #ecf0f1;
            }
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
                border-radius: 4px;
            }
            QListWidget::item:hover {
                background-color: #d6eaf8;
            }
        """)
        self.video_list.currentItemChanged.connect(self.video_selected)
        list_container.addWidget(self.video_list)
        left_panel.addLayout(list_container)

        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setFixedWidth(300)
        left_widget.setStyleSheet("background-color: white; border-radius: 8px;")
        main_layout.addWidget(left_widget)

        # Права панель (відео + статистика)
        right_panel = QVBoxLayout()
        right_panel.setContentsMargins(0, 10, 10, 10)

        # Створення прокручуваної області для правої панелі
        right_scroll_area = QScrollArea()
        right_scroll_area.setWidgetResizable(True)
        right_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll_area.setStyleSheet("border: none;")

        right_scroll_widget = QWidget()
        right_scroll_layout = QVBoxLayout(right_scroll_widget)
        right_scroll_layout.setSpacing(15)

        # Контейнер для відео
        self.video_container = QWidget()
        self.video_container.setStyleSheet("background-color: white; border-radius: 8px; padding: 10px;")
        video_layout = QVBoxLayout(self.video_container)

        # Заголовок відео
        video_title = QLabel('Перегляд відео')
        video_title.setStyleSheet('font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #34495e;')
        video_layout.addWidget(video_title)

        # Відеоплеєр
        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.error.connect(self.handle_player_error)
        self.media_player.stateChanged.connect(self.media_state_changed)

        # Відеовіджет
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(360)
        self.video_widget.setStyleSheet("background-color: #2c3e50; border-radius: 4px;")
        self.media_player.setVideoOutput(self.video_widget)
        video_layout.addWidget(self.video_widget)

        # Елементи керування відео
        player_controls = QHBoxLayout()
        player_controls.setContentsMargins(0, 10, 0, 10)

        self.play_button = QPushButton('Відтворити')
        self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.play_button.clicked.connect(self.play_video)
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                font-size: 14px;
                padding: 8px 16px;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
        """)
        player_controls.addWidget(self.play_button)

        self.pause_button = QPushButton('Пауза')
        self.pause_button.setIcon(QIcon.fromTheme("media-playback-pause"))
        self.pause_button.clicked.connect(self.pause_video)
        self.pause_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-size: 14px;
                padding: 8px 16px;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        player_controls.addWidget(self.pause_button)

        player_controls.addSpacing(20)
        player_controls.addStretch()

        video_layout.addLayout(player_controls)

        # Статус відтворення
        status_container = QHBoxLayout()
        status_icon = QLabel()
        status_icon.setPixmap(QIcon.fromTheme("dialog-information").pixmap(16, 16))
        status_container.addWidget(status_icon)

        self.player_status = QLabel('Відео не завантажено')
        self.player_status.setStyleSheet("color: #7f8c8d; font-style: italic;")
        status_container.addWidget(self.player_status)
        status_container.addStretch()

        video_layout.addLayout(status_container)

        # Прогрес обробки
        progress_container = QVBoxLayout()
        self.progress_label = QLabel('Відео обробляється, будь ласка зачекайте...')
        self.progress_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        progress_container.addWidget(self.progress_label)
        self.progress_label.hide()
        video_layout.addLayout(progress_container)

        # Додаємо відео контейнер до прокручуваної області
        right_scroll_layout.addWidget(self.video_container)

        # Створюємо контейнер для статистики
        stats_widget = QWidget()
        stats_widget.setStyleSheet("background-color: white; border-radius: 8px; padding: 15px;")
        self.stats_container = QVBoxLayout(stats_widget)

        # Заголовок статистики
        stats_title = QLabel('Статистика матчу')
        stats_title.setStyleSheet('font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #34495e;')
        self.stats_container.addWidget(stats_title)

        # Контейнер для володіння м'ячем
        possession_container = QWidget()
        possession_container.setStyleSheet("background-color: #ecf0f1; border-radius: 6px; padding: 10px;")
        possession_layout = QVBoxLayout(possession_container)

        possession_title = QLabel('Володіння м\'ячем:')
        possession_title.setStyleSheet('font-size: 16px; font-weight: bold; margin-bottom: 10px; color: #2c3e50;')
        possession_layout.addWidget(possession_title)

        self.team1_layout, self.team1_color, self.team1_possession = self.create_team_layout('Команда 1:')
        possession_layout.addLayout(self.team1_layout)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #bdc3c7; margin: 5px 0;")
        possession_layout.addWidget(separator)

        self.team2_layout, self.team2_color, self.team2_possession = self.create_team_layout('Команда 2:')
        possession_layout.addLayout(self.team2_layout)

        self.stats_container.addWidget(possession_container)

        # Заголовок для статистики гравців
        players_title = QLabel('Статистика гравців:')
        players_title.setStyleSheet(
            'font-size: 16px; font-weight: bold; margin-top: 15px; margin-bottom: 10px; color: #2c3e50;')
        self.stats_container.addWidget(players_title)

        # Контейнер для статистики гравців - прибрано окрему прокрутку
        players_container = QWidget()
        players_container.setStyleSheet("background-color: #ecf0f1; border-radius: 6px; padding: 10px;")
        self.players_stats_container = QVBoxLayout(players_container)
        self.players_stats_container.setSpacing(10)  # Збільшуємо відстань між елементами

        # Додаємо контейнер безпосередньо до stats_container
        self.stats_container.addWidget(players_container)

        # Додаємо віджет статистики до прокручуваної області
        right_scroll_layout.addWidget(stats_widget)

        # Встановлюємо остаточну прокручувану область до правої панелі
        right_scroll_area.setWidget(right_scroll_widget)
        right_panel.addWidget(right_scroll_area)

        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        main_layout.addWidget(right_widget)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def create_team_layout(self, team_name):
        layout = QHBoxLayout()

        team_info = QHBoxLayout()
        label = QLabel(team_name)
        label.setStyleSheet('font-weight: bold; min-width: 100px;')

        color_frame = QFrame()
        color_frame.setFixedSize(20, 20)
        color_frame.setStyleSheet("border: 1px solid #bdc3c7; border-radius: 3px;")

        team_info.addWidget(label)
        team_info.addWidget(color_frame)
        team_info.addStretch()
        layout.addLayout(team_info, 1)

        possession_layout = QHBoxLayout()
        possession_label = QLabel('0%')
        possession_label.setStyleSheet('font-size: 16px; font-weight: bold; color: #2980b9;')
        possession_layout.addWidget(possession_label)
        possession_layout.addStretch()
        layout.addLayout(possession_layout, 1)

        return layout, color_frame, possession_label

    def upload_video(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, 'Завантажити відео', '', 'Video Files (*.mp4 *.avi *.mov *.mkv)'
        )

        if file_path:
            self.progress_label.show()
            self.progress_label.setText(f'Обробка відео: {os.path.basename(file_path)}...')

            self.processing_thread = VideoProcessThread(file_path)
            self.processing_thread.finished.connect(self.video_processed)
            self.processing_thread.start()

    def video_processed(self, output_path, stats):
        self.progress_label.hide()

        if output_path:
            video_name = os.path.basename(output_path)

            if not os.path.exists(output_path):
                self.show_error_message(f'Файл {output_path} не знайдено!')
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
                    self.show_error_message(f'Файл {self.current_video} не знайдено!')
                    return

                print(f'Завантаження відео: {self.current_video}')
                self.load_video(self.current_video)
                self.display_statistics(video_data['stats'])

    def show_error_message(self, message):
        self.player_status.setText(f'ПОМИЛКА: {message}')
        self.player_status.setStyleSheet("color: #e74c3c; font-weight: bold;")

    def load_video(self, video_path):
        abs_path = os.path.abspath(video_path)
        print(f'Абсолютний шлях до відео: {abs_path}')

        if not os.path.exists(abs_path):
            self.show_error_message(f'Файл не знайдено: {abs_path}')
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
            self.player_status.setStyleSheet("color: #f39c12; font-style: italic;")
            return

        self.media_player.play()
        self.player_status.setText('Відтворення...')
        self.player_status.setStyleSheet("color: #27ae60; font-weight: bold;")

    def pause_video(self):
        self.media_player.pause()
        self.player_status.setText('Пауза')
        self.player_status.setStyleSheet("color: #7f8c8d; font-style: italic;")

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
        self.show_error_message(f'Помилка відтворення: {error_text}')
        print(f'Помилка плеєра: {error_text}')

    def media_state_changed(self, state):
        states = {
            QMediaPlayer.StoppedState: 'Зупинено',
            QMediaPlayer.PlayingState: 'Відтворення',
            QMediaPlayer.PausedState: 'Пауза'
        }

        status_text = states.get(state, f'Невідомий стан: {state}')
        self.player_status.setText(status_text)

        if state == QMediaPlayer.PlayingState:
            self.player_status.setStyleSheet("color: #27ae60; font-weight: bold;")
        else:
            self.player_status.setStyleSheet("color: #7f8c8d; font-style: italic;")

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
                    self.team1_color.setStyleSheet(
                        f'background-color: rgb({color[0]}, {color[1]}, {color[2]}); '
                        f'border: 1px solid #bdc3c7; border-radius: 3px;'
                    )

                if 2 in team_colors:
                    color = team_colors[2]
                    self.team2_color.setStyleSheet(
                        f'background-color: rgb({color[0]}, {color[1]}, {color[2]}); '
                        f'border: 1px solid #bdc3c7; border-radius: 3px;'
                    )

        if 'players' in stats:
            for player_id, player_stats in stats['players'].items():
                player_card = QFrame()
                player_card.setStyleSheet("""
                    background: white;
                    border-radius: 4px;
                    padding: 5px;
                    margin-bottom: 8px;
                """)
                player_layout = QVBoxLayout(player_card)

                header_layout = QHBoxLayout()

                team_color_frame = QFrame()
                team_color_frame.setFixedSize(15, 15)
                if 'team_color' in player_stats:
                    color = player_stats['team_color']
                    team_color_frame.setStyleSheet(
                        f'background-color: rgb({color[0]}, {color[1]}, {color[2]}); border-radius: 7px;')
                header_layout.addWidget(team_color_frame)

                player_id_label = QLabel(f'Гравець {player_id}')
                player_id_label.setStyleSheet('font-weight: bold; font-size: 14px;')
                header_layout.addWidget(player_id_label)
                header_layout.addStretch()

                player_layout.addLayout(header_layout)

                stats_layout = QHBoxLayout()

                speed_container = QVBoxLayout()
                speed_title = QLabel('Швидкість')
                speed_title.setStyleSheet('color: #7f8c8d; font-size: 12px;')
                speed_container.addWidget(speed_title)

                avg_speed = player_stats.get('avg_speed', 0)
                speed_value = QLabel(f'{avg_speed:.2f} км/год')
                speed_value.setStyleSheet('font-weight: bold; font-size: 14px; color: #2980b9;')
                speed_container.addWidget(speed_value)
                stats_layout.addLayout(speed_container)

                line = QFrame()
                line.setFrameShape(QFrame.VLine)
                line.setFrameShadow(QFrame.Sunken)
                line.setStyleSheet('background-color: #ecf0f1;')
                stats_layout.addWidget(line)

                distance_container = QVBoxLayout()
                distance_title = QLabel('Дистанція')
                distance_title.setStyleSheet('color: #7f8c8d; font-size: 12px;')
                distance_container.addWidget(distance_title)

                total_distance = player_stats.get('total_distance', 0)
                distance_value = QLabel(f'{total_distance:.2f} м')
                distance_value.setStyleSheet('font-weight: bold; font-size: 14px; color: #27ae60;')
                distance_container.addWidget(distance_value)
                stats_layout.addLayout(distance_container)

                player_layout.addLayout(stats_layout)

                self.players_stats_container.addWidget(player_card)

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
