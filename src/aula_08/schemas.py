from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel

CONTENT_TYPE = List[Dict[str, Union[str, Dict[str, str]]]]


class AzureLogHandler(BaseModel):
    connection_string: str


class SourceEngineSchema(BaseModel):
    origin: Dict[str, Any]
    destination: Dict[str, Any]


class AzureAIMessage(BaseModel):
    role: Literal["system", "user"]
    content: str
    name: Optional[str] = None


class AzureDateSource(BaseModel):
    type: str
    parameters: Dict[str, Any]


class AzureAIFunction(BaseModel):
    name: str
    description: Optional[str]
    parameters: Dict[str, Any]


class AzureAITool(BaseModel):
    type: str
    function: AzureAIFunction

