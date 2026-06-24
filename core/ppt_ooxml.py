"""
PPT 内 OOXML 文本节点枚举与写回。
解析与生成必须使用相同的遍历规则（文档序、相同标签过滤），否则序号会错位。
"""
from __future__ import annotations

import errno
import os
import re
import shutil
import tempfile
from collections import defaultdict
from typing import Dict, List, Tuple

from lxml import etree
from zipfile import ZipFile, ZIP_DEFLATED

# DrawingML 主命名空间（<a:t> 实际标签名为 {ns}t）
PPTX_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
# 图表
PPTX_C_NS = "http://schemas.openxmlformats.org/drawingml/2006/chart"

_TAG_AT = f"{{{PPTX_A_NS}}}t"
_TAG_CV = f"{{{PPTX_C_NS}}}v"


def _chart_v_translatable(s: str) -> bool:
    """排除纯数值刻度等，仅保留含字母的 c:v（标题类缓存文本）。"""
    t = s.strip()
    if not t:
        return False
    if not any(ch.isalpha() for ch in t):
        return False
    num = t.replace(",", "").replace("%", "").strip()
    if re.fullmatch(r"[-+]?\d*\.?\d+([eE][-+]?\d+)?", num):
        return False
    return True


def ppt_ooxml_list_writable_text_nodes(root: etree._Element, part_path: str) -> List[etree._Element]:
    """
    按文档顺序返回可改写 .text 的元素列表（与解析时一致）。
    - 所有部件中的 DrawingML 文本 run：a:t
    - 图表部件中额外包含符合条件的 c:v
    """
    norm = part_path.replace("\\", "/")
    is_chart = "/charts/chart" in norm and norm.endswith(".xml")
    out: List[etree._Element] = []
    for el in root.iter():
        if el.tag == _TAG_AT:
            tx = (el.text or "").strip()
            if tx:
                out.append(el)
        elif is_chart and el.tag == _TAG_CV:
            tx = (el.text or "").strip()
            if tx and _chart_v_translatable(tx):
                out.append(el)
    return out


def ppt_ooxml_ordered_part_paths(zf: ZipFile) -> List[str]:
    """diagrams -> charts -> notesSlides，各组内按路径排序，保证稳定顺序。"""
    names = zf.namelist()
    diagrams = sorted(
        n for n in names if n.startswith("ppt/diagrams/") and n.endswith(".xml")
    )
    charts = sorted(
        n
        for n in names
        if n.startswith("ppt/charts/") and n.lower().endswith(".xml")
    )
    notes = sorted(
        n
        for n in names
        if n.startswith("ppt/notesSlides/notesSlide") and n.endswith(".xml")
    )
    return diagrams + charts + notes


def apply_ppt_ooxml_translations(
    pptx_path: str, updates: List[Tuple[str, int, str]]
) -> int:
    """
    就地修补 pptx（Zip）内 XML 文本节点。

    Args:
        pptx_path: 已保存的 .pptx 路径
        updates: (part_path, t_index, new_text)

    Returns:
        成功写入的节点数
    """
    if not updates:
        return 0

    by_part: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for part, idx, text in updates:
        by_part[part].append((idx, text))
    for part in by_part:
        by_part[part].sort(key=lambda x: x[0])

    written = 0
    # 临时文件必须与目标在同一盘符：Windows 上 os.replace 不能跨卷（WinError 17）
    dest_dir = os.path.dirname(os.path.abspath(pptx_path)) or "."
    try:
        fd, temp_path = tempfile.mkstemp(suffix=".pptx", prefix="~ppt_ooxml_", dir=dest_dir)
    except OSError:
        fd, temp_path = tempfile.mkstemp(suffix=".pptx", prefix="~ppt_ooxml_")
    os.close(fd)
    try:
        with ZipFile(pptx_path, "r") as zin:
            with ZipFile(
                temp_path, "w", compression=ZIP_DEFLATED, compresslevel=6
            ) as zout:
                for item in zin.infolist():
                    raw = zin.read(item.filename)
                    if item.filename in by_part:
                        root = etree.fromstring(raw)
                        nodes = ppt_ooxml_list_writable_text_nodes(
                            root, item.filename
                        )
                        for t_idx, new_text in by_part[item.filename]:
                            if 0 <= t_idx < len(nodes):
                                nodes[t_idx].text = new_text
                                written += 1
                        raw = etree.tostring(
                            root,
                            xml_declaration=True,
                            encoding="UTF-8",
                            standalone=None,
                        )
                    zout.writestr(item, raw)
        try:
            os.replace(temp_path, pptx_path)
        except OSError as exc:
            # 跨卷或其它 rename 失败：复制覆盖目标再删临时文件
            winerr = getattr(exc, "winerror", None)
            exdev = getattr(errno, "EXDEV", None)
            if winerr == 17 or (exdev is not None and exc.errno == exdev):
                shutil.copyfile(temp_path, pptx_path)
                os.unlink(temp_path)
            else:
                raise
    except Exception:
        if os.path.isfile(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        raise
    return written
