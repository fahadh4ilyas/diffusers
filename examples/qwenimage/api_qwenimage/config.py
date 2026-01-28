import dotenv, logging, typing
from pydantic import Field, field_validator, ValidationInfo
try:
    from pydantic import BaseSettings
except:
    from pydantic_settings import BaseSettings

class ApiConfig(BaseSettings):
    api_host: str = Field('127.0.0.1')
    api_port: int = Field(5000)
    worker_num: int = Field(1)
    forwarded_allow_ips: str = Field('127.0.0.1')
    model_path: str = Field(...)
    force_lora: bool = Field(False)
    lora_path: typing.Optional[str] = Field(None)
    lora_keywords: typing.Optional[str] = Field(None)
    include_keywords: bool = Field(False)
    max_batch_size: int = Field(4)
    image_input: bool = Field(False)

    class Config:
        env_file = dotenv.find_dotenv(usecwd=True)
        env_file_encoding = 'utf-8'
        extra = 'ignore'

    @field_validator('lora_path')
    @classmethod
    def validate_lora_path(cls, v: typing.Optional[str], info: ValidationInfo) -> typing.Optional[str]:

        if not v and info.data['force_lora']:
            raise ValueError('`lora_path` field is required if `force_lora` is True.')
        return v
    
    @field_validator('lora_keywords')
    @classmethod
    def validate_lora_keywords(cls, v: typing.Optional[str], info: ValidationInfo) -> typing.Optional[str]:

        if not v and info.data['lora_path'] and not info.data['force_lora']:
            raise ValueError('`lora_keywords` field is required if `lora_path` exists and `force_lora` is False.')
        return v

config = ApiConfig()


LOGGER = logging.getLogger('gunicorn.error')
LOGGER_ACCESS = logging.getLogger('gunicorn.access')