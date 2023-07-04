from dataclasses import dataclass
from datetime import datetime


@dataclass
class Page:
    url: str
    domain: str
    date: datetime
    html: str
