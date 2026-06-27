import re


TITLE_PATTERN = re.compile(r"【[^】]+】")
ORDER_PATTERN = re.compile(r"ORDER\d+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？；])")


def normalize_text(text: str) -> str:
    """
    保留段落边界，只清理空行和每行两侧空白。
    """
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]
    return "\n".join(lines)


def split_by_title(text: str) -> list[str]:
    """
    第一层：按 【标题】 切块。
    """
    text = normalize_text(text)

    parts = re.split(r"\n?(?=【[^】]+】)", text)

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def extract_order_id(block: str) -> str | None:
    """
    提取订单号，例如 ORDER1001。
    用于判断相邻块是否属于同一个订单。
    """
    match = ORDER_PATTERN.search(block)
    if not match:
        return None

    return match.group(0)


def merge_same_order_blocks(
    blocks: list[str],
    max_chars: int = 600,
) -> list[str]:
    """
    第二层：同一个订单的相邻块可以合并。
    """
    merged: list[str] = []

    buffer = ""
    buffer_order_id: str | None = None

    for block in blocks:
        current_order_id = extract_order_id(block)

        if not buffer:
            buffer = block
            buffer_order_id = current_order_id
            continue

        can_merge = (
            buffer_order_id is not None
            and current_order_id == buffer_order_id
            and len(buffer) + len(block) <= max_chars
        )

        if can_merge:
            buffer = f"{buffer}\n{block}"
        else:
            merged.append(buffer)
            buffer = block
            buffer_order_id = current_order_id

    if buffer:
        merged.append(buffer)

    return merged


def split_long_block(
    block: str,
    max_chars: int = 600,
) -> list[str]:
    """
    第三层：如果某个结构块太长，再按句子切。
    如果单句也特别长，再用固定长度兜底。
    """
    if len(block) <= max_chars:
        return [block]

    sentences = [
        sentence.strip()
        for sentence in SENTENCE_SPLIT_PATTERN.split(block)
        if sentence.strip()
    ]

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""

            for start in range(0, len(sentence), max_chars):
                chunks.append(sentence[start:start + max_chars])
            continue

        if not current:
            current = sentence
            continue

        if len(current) + len(sentence) <= max_chars:
            current += sentence
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def chunk_text(
    text: str,
    max_chars: int = 600,
    merge_same_order: bool = True,
) -> list[str]:
    """
    对外统一暴露的切块函数。

    策略：
    1. 优先按 【标题】 切块
    2. 相邻同订单块合并
    3. 超长块按句子兜底切分
    """
    blocks = split_by_title(text)

    if merge_same_order:
        blocks = merge_same_order_blocks(
            blocks=blocks,
            max_chars=max_chars,
        )

    chunks: list[str] = []

    for block in blocks:
        chunks.extend(
            split_long_block(
                block=block,
                max_chars=max_chars,
            )
        )

    return chunks