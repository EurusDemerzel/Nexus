import os

import pymysql
from dotenv import load_dotenv


load_dotenv()

def populate_database():
    # 数据库连接配置（优先读取 .env）
    db_config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", "123456"),
        "database": os.getenv("DB_NAME", "patent_db"),
        "charset": "utf8mb4",
        "cursorclass": pymysql.cursors.Cursor,
    }

    # 连接到数据库
    connection = pymysql.connect(**db_config)
    cursor = connection.cursor()

    # 示例数据
    data = [
        ('课程标题1', '这是课程内容1，包含中文信息。'),
        ('课程标题2', '这是课程内容2，包含中文信息。'),
        ('课程标题3', '这是课程内容3，包含中文信息。'),
    ]

    # 插入数据到数据库
    insert_query = "INSERT INTO patents (title, content) VALUES (%s, %s)"
    cursor.executemany(insert_query, data)

    # 提交事务
    connection.commit()

    # 关闭连接
    cursor.close()
    connection.close()

if __name__ == "__main__":
    populate_database()