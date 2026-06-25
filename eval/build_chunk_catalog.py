import json
from pathlib import Path


def get_project_root() -> Path:
    """
    基于当前脚本位置定位项目根目录。
    避免依赖你在哪个目录运行 python 命令。
    """
    return Path(__file__).resolve().parents[1]


def load_docs(data_dir: Path) -> dict[str, str]:
    """
    读取 test 目录下的 txt 文件。
    key 是文件名 stem，例如 study/life/thinking。
    value 是文件内容。
    """
    docs = {}

    for file_path in data_dir.glob("*.txt"):
        category = file_path.stem
        docs[category] = file_path.read_text(encoding="utf-8")

    return docs


def chunk_text(
    text: str,
    chunk_size: int = 150,
    chunk_overlap: int = 40,
) -> list[str]:
    """
    必须和 ingest.py 当前入库切块逻辑保持一致。
    这里先复制一份，后面可以再抽到公共模块。
    """
    chunks = []
    start = 0

    text = "".join(
        line.strip()
        for line in text.splitlines()
        if line.strip()
    )

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - chunk_overlap

        if start >= len(text) or end >= len(text):
            break

    return chunks


def build_chunk_catalog() -> None:
    project_root = get_project_root()
    data_dir = project_root / "test"
    output_path = project_root / "eval" / "chunk_catalog.json"

    docs = load_docs(data_dir)

    catalog = []

    for category, raw_text in docs.items():
        chunks = chunk_text(raw_text, chunk_size=150, chunk_overlap=40)

        for index, content in enumerate(chunks):
            catalog.append({
                "chunk_id": f"{category}_{index}",
                "category": category,
                "source": f"{category}.txt",
                "content": content,
            })

    output_path.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"chunk catalog generated: {output_path}")
    print(f"total chunks: {len(catalog)}")


if __name__ == "__main__":
    build_chunk_catalog()