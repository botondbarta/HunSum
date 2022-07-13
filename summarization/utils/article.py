from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class Article:
    title: str
    lead: str
    article: str
    domain: str
    url: str
    date: datetime

    def to_json(self) -> str:
        return json.dumps(self.__dict__)


