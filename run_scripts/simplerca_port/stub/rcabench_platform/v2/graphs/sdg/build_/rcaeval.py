# load_inject_time copied verbatim from rcabench_platform 0.3.27 v2/graphs/sdg/build_/rcaeval.py (logger call dropped).
import datetime
from pathlib import Path


def load_inject_time(input_folder: Path) -> datetime.datetime:
    inject_time_file = input_folder / "inject_time.txt"

    with open(inject_time_file) as f:
        timestamp = int(f.read().strip())
    inject_time = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
    return inject_time
