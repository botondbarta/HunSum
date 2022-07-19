import json
from dataclasses import dataclass
from datetime import datetime
from typing import Set

from summarization.serializers.datetime_serializer import DateTimeEncoder


@dataclass
class Article:
    title: str
    lead: str
    article: str
    domain: str
    url: str
    date: datetime
    tags: Set[str]

    def to_json(self) -> str:
        return json.dumps(self.__dict__, cls=DateTimeEncoder)


