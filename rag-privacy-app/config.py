import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DEBUG = True
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_default_secret_key'
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'mysql+pymysql://username:password@localhost/db_name'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # LLM API configuration
    LLM_API_KEY = os.getenv("LLM_API_KEY")

    # Other configurations can be added here as needed
    # For example, you can add configurations for caching, logging, etc.