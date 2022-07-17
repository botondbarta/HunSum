from datetime import datetime
from json import JSONEncoder


class DateTimeEncoder(JSONEncoder):
    # Override the default method
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
