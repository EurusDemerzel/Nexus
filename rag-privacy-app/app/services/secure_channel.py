# app/services/secure_channel.py
from cryptography.fernet import Fernet
import pickle
import os

# 从环境变量读取密钥，如果没有则生成一个固定的（仅演示，生产环境应妥善保存）
KEY = os.environ.get('FERNET_KEY', b'3Vp8Z8RgVl1HvDq6Tq5Gt7eL9jWkXcFbNnMpLkIjUhY=')
if isinstance(KEY, str):
    KEY = KEY.encode()
cipher = Fernet(KEY)

def encrypt_object(obj):
    data = pickle.dumps(obj)
    return cipher.encrypt(data)

def decrypt_object(encrypted):
    data = cipher.decrypt(encrypted)
    return pickle.loads(data)