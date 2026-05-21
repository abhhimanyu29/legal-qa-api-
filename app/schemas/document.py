from pydantic import BaseModel
from datetime import datetime


class DocumentCreate(BaseModel):
    filename: str


class DocumentResponse(BaseModel):
    id: int
    filename: str
    upload_time: datetime

    class Config:
        from_attributes = True
from datetime import datetime


class DocumentResponse(BaseModel):
    id: int
    filename: str
    upload_time: datetime

    class Config:
        from_attributes = True