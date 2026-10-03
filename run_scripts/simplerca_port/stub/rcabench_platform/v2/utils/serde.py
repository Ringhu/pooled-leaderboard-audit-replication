def save_parquet(df, path):
    raise RuntimeError("debug output disabled in stub")


def load_json(path):
    import json
    with open(path) as f:
        return json.load(f)
