import json
from dataclasses import dataclass
from datetime import datetime
from typing import List

from serializers.datetime_serializer import DateTimeEncoder


@dataclass
class Article:
    title: str
    lead: str
    article: str
    domain: str
    url: str
    date: datetime
    tags: List[str]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, cls=DateTimeEncoder)


