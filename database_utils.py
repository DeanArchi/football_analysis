import ast
import random
import uuid

import psycopg2


def get_connection():
    """
    Встановлює з'єднання з базою даних PostgreSQL.

    :return: об'єкт з'єднання
    """
    return psycopg2.connect(
        dbname="football_analysis",
        user="postgres",
        password="password",
        host="localhost",
        port=5432
    )


def insert_match(video_bytes, video_name):
    """
    Зберігає відео в таблиці match_info.

    :param video_bytes: байтовий потік відео
    :param video_name: назва відео
    :return: match_id отриманого запису
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        match_id = str(uuid.uuid4())
        insert_query = """
        INSERT INTO match_info (match_id, match_video_name, processed_match_video)
        VALUES (%s, %s, %s);
        """
        cur.execute(insert_query, (match_id, video_name, psycopg2.Binary(video_bytes)))
        conn.commit()
        return match_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()


def insert_team_and_stats(match_id, team_color, ball_possession):
    """
    Додає інформацію про команду та відповідну статистику.

    :param match_id: ідентифікатор матчу
    :param team_color: колір команди
    :param ball_possession: відсоток володіння м'ячем (ціле число)
    :return: team_id отриманої команди
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        team_id = str(uuid.uuid4())
        team_query = """
        INSERT INTO team (team_id, match_id, team_color)
        VALUES (%s, %s, %s);
        """
        cur.execute(team_query, (team_id, match_id, team_color))

        teamstats_query = """
        INSERT INTO teamstats (team_id, match_id, ball_possession)
        VALUES (%s, %s, %s);
        """
        cur.execute(teamstats_query, (team_id, match_id, ball_possession))

        conn.commit()
        return team_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()


def insert_player_and_stats(team_id, match_id, player_number, distance, avg_speed):
    """
    Додає інформацію про гравця та відповідну статистику.

    :param team_id: ідентифікатор команди
    :param match_id: ідентифікатор матчу
    :param player_number: номер гравця
    :param distance: пройдена відстань (тип float)
    :param avg_speed: середня швидкість (тип float)
    :return: player_id отриманого гравця
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        player_id = str(uuid.uuid4())
        player_query = """
        INSERT INTO player (player_id, team_id, match_id, player_number)
        VALUES (%s, %s, %s, %s);
        """
        cur.execute(player_query, (player_id, team_id, match_id, player_number))

        playerstats_query = """
        INSERT INTO playerstats (player_id, team_id, match_id, distance, avg_speed)
        VALUES (%s, %s, %s, %s, %s);
        """
        cur.execute(playerstats_query, (player_id, team_id, match_id, distance, avg_speed))

        conn.commit()
        return player_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()


def fetch_all_matches():
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT match_id, match_video_name FROM match_info;")
        rows = cur.fetchall()
        return [{'match_id': row[0], 'match_video_name': row[1]} for row in rows]
    finally:
        cur.close()
        conn.close()


def fetch_match(match_id):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT processed_match_video FROM match_info WHERE match_id = %s", (match_id,))
        row = cur.fetchone()
        return row
    finally:
        cur.close()
        conn.close()


def fetch_all_teams():
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT team_id, match_id, team_color FROM team;")
        rows = cur.fetchall()
        return [{'team_id': row[0], 'match_id': row[1], 'team_color': row[2]} for row in rows]
    finally:
        cur.close()
        conn.close()


def fetch_all_teamstats():
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT team_id, match_id, ball_possession FROM teamstats;")
        rows = cur.fetchall()
        return [{'team_id': row[0], 'match_id': row[1], 'ball_possession': row[2]} for row in rows]
    finally:
        cur.close()
        conn.close()


def fetch_all_players():
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT player_id, team_id, match_id, player_number FROM player;")
        rows = cur.fetchall()
        return [{'player_id': row[0], 'team_id': row[1], 'match_id': row[2], 'player_number': row[3]} for row in rows]
    finally:
        cur.close()
        conn.close()


def fetch_all_playerstats():
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT player_id, team_id, match_id, distance, avg_speed FROM playerstats;")
        rows = cur.fetchall()
        return [{'player_id': row[0], 'team_id': row[1], 'match_id': row[2], 'distance': row[3], 'avg_speed': row[4]} for row in rows]
    finally:
        cur.close()
        conn.close()


def fetch_team_colors(match_id):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT team_id, team_color FROM team WHERE match_id = %s", (match_id,))
    results = cur.fetchall()

    team_colors = {}
    for row in results:
        team_id, color_str = row
        try:
            rgb = tuple(int(c) for c in color_str.strip("()").split(','))
            team_colors[team_id] = rgb
        except ValueError:
            team_colors[team_id] = (0, 0, 0)

    cur.close()
    conn.close()
    return team_colors

