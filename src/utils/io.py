import json
from pathlib import Path
from typing import Any, List


def save_json(data: Any, path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_jsonl(records: List[dict], path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def load_jsonl(path: Path) -> List[dict]:

    data = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    return data
