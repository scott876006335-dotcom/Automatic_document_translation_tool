"""文本处理模块"""
import re
from typing import Dict, List, Tuple


class TextProcessor:
    """文本处理和编号类"""

    # 兼容常见编号样式：T0001 / P0043 / Page01-P0001 / S01_P001
    _ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-\.]*$")
    
    @staticmethod
    def number_texts(text_dict: Dict[str, str]) -> Dict[str, str]:
        """
        为文本字典添加编号（支持超大文件）
        
        Args:
            text_dict: 原始文本字典，格式为 {key: text}
            
        Returns:
            编号后的文本字典，格式为 {编号: {key: 原始key, text: 文本内容}}
        
        编号规则：
        - 1-9999: T0001 ~ T9999 (4位)
        - 10000-99999: T10000 ~ T99999 (5位)
        - 100000+: T100000+ (6位及以上)
        """
        numbered_texts = {}
        index = 1
        
        # 预先计算总数，确定位数
        total_count = sum(1 for text in text_dict.values() if ' '.join(text.split()))
        
        # 动态确定位数（最少4位）
        if total_count < 10000:
            digit_width = 4
        elif total_count < 100000:
            digit_width = 5
        elif total_count < 1000000:
            digit_width = 6
        else:
            digit_width = 7  # 支持到999万
        
        for key, text in text_dict.items():
            cleaned_text = ' '.join(text.split())
            if cleaned_text:
                # 使用动态位数格式化
                number = f"T{index:0{digit_width}d}"
                numbered_texts[number] = {
                    'key': key,
                    'text': cleaned_text
                }
                index += 1
        
        return numbered_texts
    
    @staticmethod
    def format_text_for_translation(numbered_texts: Dict[str, Dict]) -> str:
        """
        将文本字典格式化为待翻译字符串
        注意：使用 @staticmethod，且参数中不带 self
        """
        lines = []
        for number, data in numbered_texts.items():
            # 确保提取 text 字段，如果 data 是字符串则直接使用
            text = data['text'] if isinstance(data, dict) and 'text' in data else str(data)
            text = text.replace('\n', ' ').strip()
            # 使用冒号分隔
            lines.append(f"{number}: {text}")
        return "\n".join(lines)

    
    # 模型常把多条「ID|译文」挤在同一行，用此模式切分为多条记录（避免整行被当成一条译文）。
    _BAR_RECORD_HEAD = re.compile(r"(?:^|[\s;])([A-Za-z0-9][A-Za-z0-9\-_.]*)\|")

    @staticmethod
    def _translation_id_valid(potential_id: str) -> bool:
        if not potential_id:
            return False
        return bool(
            re.match(r"^[A-Za-z][0-9]{4,}$", potential_id)
            or TextProcessor._ID_PATTERN.match(potential_id)
        )

    @staticmethod
    def _split_line_bar_records(line: str) -> List[Tuple[str, str]]:
        """若一行含多个「编号|」，拆成 [(id, text), ...]；否则返回空列表表示走单行解析。"""
        matches = list(TextProcessor._BAR_RECORD_HEAD.finditer(line))
        if len(matches) <= 1:
            return []
        out: List[Tuple[str, str]] = []
        for i, m in enumerate(matches):
            key = m.group(1)
            if not TextProcessor._translation_id_valid(key):
                return []
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(line)
            text = line[start:end].strip()
            out.append((key, text))
        return out

    @staticmethod
    def parse_translation_result(result_text: str) -> Dict[str, str]:
        """
        解析翻译结果（支持任意长度编号）
        """
        translations = {}
        lines = result_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 清理干扰字符
            line = line.replace('**', '').replace('`', '')
            
            # 跳过明显非翻译行的内容
            if line.lower().startswith(('here is', 'translation:', 'sure', '请', '要求')):
                continue

            merged_records = TextProcessor._split_line_bar_records(line)
            if merged_records:
                for number, translation in merged_records:
                    if translation:
                        translations[number] = translation
                continue
            
            number = None
            translation = None
            
            # --- 策略 1：竖线分割 ---
            if '|' in line:
                parts = [p.strip() for p in line.split('|')]
                
                # 处理模型添加额外序号的情况
                if len(parts) > 2 and parts[0].isdigit():
                    parts = parts[1:]
                
                if len(parts) >= 2:
                    potential_id = parts[0]
                    # 支持大小写混合编号（如 Page01-P0001）
                    if TextProcessor._translation_id_valid(potential_id):
                        number = potential_id
                        translation = '|'.join(parts[1:])
            
            # --- 策略 2：点号分割 ---
            if not number and '.' in line:
                parts = line.split('.', 1)
                potential_id = parts[0].strip()
                if potential_id.isdigit():
                    number = potential_id
                    translation = parts[1].strip()

            # --- 策略 3：冒号分割 ---
            if not number and ':' in line:
                parts = line.split(':', 1)
                potential_id = parts[0].strip()
                # 支持大小写混合编号（如 Page01-P0001）
                if ' ' not in potential_id and (
                    re.match(r"^[A-Za-z][0-9]{4,}$", potential_id)
                    or TextProcessor._ID_PATTERN.match(potential_id)
                ):
                    number = potential_id
                    translation = parts[1].strip()

            # --- 策略 4：全角冒号（模型偶发输出「ID：译文」单行）---
            if not number and "：" in line:
                parts = line.split("：", 1)
                potential_id = parts[0].strip()
                if " " not in potential_id and TextProcessor._translation_id_valid(
                    potential_id
                ):
                    number = potential_id
                    translation = parts[1].strip()

            # --- 保存结果 ---
            if number and translation:
                translations[number] = translation
        
        return translations

    @staticmethod
    def smart_match_translations(parsed_translations: Dict[str, str], 
                                expected_keys: List[str]) -> Dict[str, str]:
        """
        智能匹配翻译结果到期望的编号
        
        支持的匹配策略：
        1. 精确匹配：P0121 -> P0121
        2. 模糊匹配：前缀/后缀及末段数字一致（如 S01-P0121 与 P0121）
        不做顺序“保底”映射，避免缺行时把错误译文填到错误编号上。
        
        Args:
            parsed_translations: 解析出的翻译结果 {模型输出的编号: 译文}
            expected_keys: 期望的编号列表 [原始文档的编号]
        
        Returns:
            匹配后的翻译结果 {原始编号: 译文}
        """
        matched_translations = {}
        used_parsed_keys = set()
        
        # === 第一轮：精确匹配 ===
        for expected_key in expected_keys:
            if expected_key in parsed_translations:
                matched_translations[expected_key] = parsed_translations[expected_key]
                used_parsed_keys.add(expected_key)
        
        # === 第二轮：模糊匹配（前缀/后缀） ===
        remaining_expected = [k for k in expected_keys if k not in matched_translations]
        remaining_parsed = {k: v for k, v in parsed_translations.items() if k not in used_parsed_keys}
        
        for expected_key in remaining_expected:
            best_match = None
            
            for parsed_key in remaining_parsed.keys():
                # 策略A：模型添加了前缀 (S01-P0121 匹配 P0121)
                if parsed_key.endswith('-' + expected_key) or parsed_key.endswith('_' + expected_key):
                    best_match = parsed_key
                    break
                
                # 策略B：模型去掉了前缀 (P0121 匹配 S01-P0121)
                if expected_key.endswith('-' + parsed_key) or expected_key.endswith('_' + parsed_key):
                    best_match = parsed_key
                    break
                
                # 策略C：提取数字部分匹配 (P0121 和 S01-P0121 都提取 0121)
                expected_num = re.findall(r'\d+', expected_key)
                parsed_num = re.findall(r'\d+', parsed_key)
                if expected_num and parsed_num and expected_num[-1] == parsed_num[-1]:
                    # 确保字母前缀也匹配（P 对 P，T 对 T）
                    expected_prefix = re.match(r'^([A-Z]+)', expected_key)
                    parsed_prefix = re.match(r'^.*?([A-Z]+)', parsed_key)
                    if expected_prefix and parsed_prefix:
                        if expected_prefix.group(1) == parsed_prefix.group(1):
                            best_match = parsed_key
                            break
            
            if best_match:
                matched_translations[expected_key] = remaining_parsed[best_match]
                used_parsed_keys.add(best_match)
                del remaining_parsed[best_match]
        
        return matched_translations

    @staticmethod
    def create_translation_prompt(
        formatted_text: str,
        source_lang_english: str,
        target_lang_english: str,
        *,
        domain_instruction: str = "",
        glossary_instruction: str = "",
    ) -> str:
        """
        构建提示词（Few-Shot，按源语言/目标语言动态生成）
        domain_instruction / glossary_instruction 为英文说明，空则省略对应块。
        """
        extra_blocks = []
        di = (domain_instruction or "").strip()
        gi = (glossary_instruction or "").strip()
        if di:
            extra_blocks.append(f"DOMAIN FOCUS:\n{di}")
        if gi:
            extra_blocks.append(gi)
        extra = ""
        if extra_blocks:
            extra = "\n\n" + "\n\n".join(extra_blocks) + "\n\n"

        return f"""You are a professional translation engine. Translate each line from {source_lang_english} into {target_lang_english}.
Preserve technical meaning; keep proper nouns, product names, standards, and code identifiers understandable (you may keep short acronyms or code as-is when translation would harm clarity).{extra}STRICT FORMAT RULES:
1. Each output line MUST be: ID|translated text only (use a single ASCII vertical bar | as separator).
2. The ID must match the input exactly (same letters, digits, hyphens, underscores, dots).
3. Output ONLY the translated lines in that format. No preface, no markdown fences, no explanations.
4. Every input line has the form "ID: text"; you output "ID|translation".
5. EXACTLY one output line per input line. NEVER put two or more "ID|..." pairs on the same line (no concatenation).
6. Do not merge or skip input lines; preserve the same ID sequence as the input block.
7. Each output line translates ONLY the text after the colon on THAT same input line. Never merge meaning from the next or previous input line into the current line (fragments and short lines stay separate).

FORMAT EXAMPLE (language below is only for demonstrating "ID|text"; your real output must be entirely in {target_lang_english}):
[INPUT]
S01-P001: Introduction to Rail Transit
P0043: The scope of the document is to cover the following topics:

[OUTPUT]
S01-P001|Introduction au transport ferroviaire urbain
P0043|La portée du document est de couvrir les sujets suivants :

DATA TO TRANSLATE (from {source_lang_english} to {target_lang_english}):
{formatted_text}

[OUTPUT]
"""

    @staticmethod
    def create_prompt(formatted_text: str) -> str:
        """兼容旧调用：默认英译简中。"""
        return TextProcessor.create_translation_prompt(
            formatted_text, "English", "Simplified Chinese"
        )

