from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Legal QA API"
    APP_VERSION: str = "1.0.0"
    OPENAI_API_KEY: str
    DEBUG: bool = False                    # ← add this
    DATABASE_URL: str = ""                 # ← add this

    class Config:
        env_file = ".env"

settings = Settings()