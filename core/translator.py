"""翻译服务模块"""
import re
import time
import subprocess
import json
import os
import multiprocessing
import urllib.error
import urllib.request
from typing import Dict, List, Callable, Optional
from utils.exceptions import LLMServiceError
from utils.logger import setup_logger
from utils.config import Config
from utils.languages import english_name_for_code
from utils.industry import build_domain_instruction_english
from utils.glossary import load_glossary_file

logger = setup_logger()


def normalize_lm_studio_base_url(raw: str) -> str:
    s = (raw or "").strip().rstrip("/")
    if not s:
        s = "http://127.0.0.1:1234"
    if s.endswith("/v1"):
        return s
    return f"{s}/v1"


class LLMTranslator:
    """本地大模型翻译（LM Studio OpenAI 兼容 API / Ollama）"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.backend = self.config.get("llm.backend", "lm_studio")
        self.model = (
            self.config.get("llm.model")
            or self.config.get("ollama.model", "qwen3.5-35b-a3b")
        )
        self.base_url = self.config.get("ollama.base_url", "http://localhost:11434")
        self.timeout = self.config.get("ollama.timeout", 300)
        self.lm_studio_socket_timeout = self._normalize_socket_timeout(
            self.config.get("lm_studio.timeout")
        )
        self.max_tokens = self.config.get("translation.max_tokens_per_batch", 32768)
        self.retry_times = self.config.get("translation.retry_times", 3)
        self.retry_delay = self.config.get("translation.retry_delay", 2)

        self.lm_studio_base_url = normalize_lm_studio_base_url(
            self.config.get("lm_studio.base_url", "http://127.0.0.1:1234/v1")
        )
        self.lm_studio_api_key = self.config.get("lm_studio.api_key", "") or ""
        self.lm_enable_thinking = self.config.get("lm_studio.enable_thinking", False)
        self.lm_send_chat_template_kwargs = self.config.get(
            "lm_studio.send_chat_template_kwargs", False
        )
        self.lm_fallback_max_tokens = self.config.get(
            "lm_studio.fallback_max_tokens", 4096
        )

        try:
            self.system_cores = multiprocessing.cpu_count()
        except Exception:
            self.system_cores = 4

        default_threads = max(1, int(self.system_cores * 0.6))
        self.num_thread = self.config.get("ollama.num_thread", default_threads)

        logger.info(
            f"LLM: backend={self.backend}, model={self.model} | "
            f"Ollama 线程优化: 核心 {self.system_cores}, num_thread={self.num_thread}"
        )
        if self.backend == "lm_studio":
            t = self.lm_studio_socket_timeout
            logger.info(
                f"LM Studio HTTP 超时: {'无限制（直到服务端关闭或出错）' if t is None else f'{t} 秒'}"
            )

    def _is_gemma4_family_model(self) -> bool:
        """Gemma 4 通过 system 中的 <|think|> 与 chat_template_kwargs.enable_thinking 控制推理。"""
        m = (self.model or "").lower().replace(" ", "")
        return "gemma-4" in m or "gemma4" in m or "gemma_4" in m

    def _build_chat_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        """OpenAI 兼容 / Ollama 共用：Gemma 4 拆成 system+user 以注入思考开关。"""
        if not self._is_gemma4_family_model():
            return [{"role": "user", "content": user_prompt}]
        if self.lm_enable_thinking:
            system = (
                "<|think|>\n"
                "You are a professional translation engine. LM Studio may record internal "
                "reasoning in message.reasoning_content; keep that concise. The user-visible "
                "answer in message.content MUST be ONLY the translation lines in the exact "
                "ID|text format described in the user message."
            )
        else:
            system = (
                "You are a professional translation engine. Thinking mode is OFF. Do not spend "
                "tokens on long internal analysis: the final assistant answer must be ONLY "
                "one line per input as ID|translated text in message.content. If the server "
                "uses a separate reasoning field (reasoning_content), keep it empty or "
                "minimal. No chain-of-thought, no \"Start thinking\" sections, no "
                "`<|channel>thought` before the ID| lines."
            )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _strip_gemma_reasoning_prefix(content: str, is_gemma4: bool) -> str:
        """若仍带出思考通道或 [Start thinking]，裁到第一条 ID| 译文行之前。"""
        if not content or not is_gemma4:
            return content
        low = content.lower()
        if "<|channel>thought" not in low and "[start thinking]" not in low:
            return content
        m = re.search(r"(?m)^[A-Za-z0-9][A-Za-z0-9\-_.]*\|", content)
        if m:
            return content[m.start() :].lstrip()
        return content

    def _lm_studio_extract_visible_text(self, msg: Dict) -> str:
        """
        使用 message.content 作为译文正文。LM Studio 对 Gemma 等模型常把推理放在
        reasoning_content（或 reasoning），与 content 分离；若 content 为空则尝试从侧栏字段
        中提取第一条 ID| 起的块。
        """
        is_gemma = self._is_gemma4_family_model()

        def _as_str(v) -> str:
            if v is None:
                return ""
            return v if isinstance(v, str) else str(v)

        content = _as_str(msg.get("content")).strip()
        if content:
            return self._strip_gemma_reasoning_prefix(content, is_gemma)

        for key in ("reasoning_content", "reasoning"):
            alt = _as_str(msg.get(key)).strip()
            if not alt:
                continue
            m = re.search(r"(?m)^[A-Za-z0-9][A-Za-z0-9\-_.]*\|", alt)
            if m:
                logger.warning(
                    "LM Studio 的 message.content 为空，已从 message.%s 提取译文块",
                    key,
                )
                return self._strip_gemma_reasoning_prefix(
                    alt[m.start() :].lstrip(), is_gemma
                )

        if msg.get("content") is not None:
            return self._strip_gemma_reasoning_prefix(
                _as_str(msg.get("content")), is_gemma
            )
        raise LLMServiceError(f"LM Studio 返回无可用正文: {msg}")

    @staticmethod
    def _normalize_socket_timeout(raw):
        """None/0/负数 表示 urllib socket 不超时；正数为秒。"""
        if raw is None:
            return None
        try:
            sec = float(raw)
        except (TypeError, ValueError):
            return None
        if sec <= 0:
            return None
        return sec

    def check_llm_available(self) -> bool:
        if self.backend == "lm_studio":
            return self._check_lm_studio_available()
        return self._check_ollama_available()

    def _check_lm_studio_available(self) -> bool:
        url = f"{self.lm_studio_base_url}/models"
        try:
            req = urllib.request.Request(url, method="GET")
            if self.lm_studio_api_key:
                req.add_header("Authorization", f"Bearer {self.lm_studio_api_key}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception as e:
            logger.warning(f"检查 LM Studio 失败: {e}")
            return False

    def _check_ollama_available(self) -> bool:
        try:
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            logger.warning(f"检查 Ollama 服务失败: {e}")
            return False

    def _call_lm_studio_api(
        self, prompt: str, progress_callback: Optional[Callable] = None
    ) -> str:
        url = f"{self.lm_studio_base_url}/chat/completions"
        messages = self._build_chat_messages(prompt)
        payload: Dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
        }
        if self.max_tokens and self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        # Qwen：开启思考或显式允许时带 kwargs；Gemma 4：必须与模板 enable_thinking 对齐，故始终传入
        if (
            self.lm_send_chat_template_kwargs
            or self.lm_enable_thinking
            or self._is_gemma4_family_model()
        ):
            payload["chat_template_kwargs"] = {
                "enable_thinking": bool(self.lm_enable_thinking)
            }
        # llama.cpp / LM Studio：0 表示不向 reasoning 通道分配预算（与侧栏 reasoning_content 对应）
        if self._is_gemma4_family_model() and not self.lm_enable_thinking:
            payload["reasoning_budget"] = 0

        logger.info(f"LM Studio 请求: {self.model} @ {self.lm_studio_base_url}")

        def post(pl: Dict) -> str:
            data = json.dumps(pl, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            if self.lm_studio_api_key:
                req.add_header("Authorization", f"Bearer {self.lm_studio_api_key}")
            with urllib.request.urlopen(req, timeout=self.lm_studio_socket_timeout) as resp:
                return resp.read().decode("utf-8")

        fallback_payload = dict(payload)
        fallback_payload.pop("chat_template_kwargs", None)
        fallback_payload.pop("reasoning_budget", None)
        fallback_max_tokens = fallback_payload.get("max_tokens")
        if (
            isinstance(fallback_max_tokens, int)
            and isinstance(self.lm_fallback_max_tokens, int)
            and self.lm_fallback_max_tokens > 0
            and fallback_max_tokens > self.lm_fallback_max_tokens
        ):
            fallback_payload["max_tokens"] = self.lm_fallback_max_tokens

        body = None
        try:
            body = post(payload)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            if e.code != 400:
                raise LLMServiceError(
                    f"LM Studio HTTP {e.code}: {err_body or e.reason}"
                ) from e
            while True:
                if payload.pop("reasoning_budget", None) is not None:
                    logger.info("LM Studio 400: 已省略 reasoning_budget 并重试")
                elif payload.pop("chat_template_kwargs", None) is not None:
                    logger.info("LM Studio 400: 已省略 chat_template_kwargs 并重试")
                else:
                    raise LLMServiceError(
                        f"LM Studio HTTP 400: {err_body or e.reason}"
                    ) from e
                try:
                    body = post(payload)
                    break
                except urllib.error.HTTPError as e2:
                    err_body = e2.read().decode("utf-8", errors="replace")
                    if e2.code != 400:
                        raise LLMServiceError(
                            f"LM Studio HTTP {e2.code}: {err_body or e2.reason}"
                        ) from e2
                    continue
        except urllib.error.URLError as e:
            reason = str(e.reason)
            if "channel error" in reason.lower() and fallback_payload != payload:
                logger.warning(
                    "LM Studio Channel Error，已使用兼容参数重试（省略可选字段并收敛 max_tokens）"
                )
                try:
                    body = post(fallback_payload)
                except Exception as e2:
                    raise LLMServiceError(
                        f"LM Studio 连接失败: {reason} | 兼容重试仍失败: {e2}"
                    ) from e2
            else:
                raise LLMServiceError(f"LM Studio 连接失败: {e.reason}") from e
        except Exception as e:
            msg = str(e)
            if "channel error" in msg.lower() and fallback_payload != payload:
                logger.warning(
                    "LM Studio Channel Error，已使用兼容参数重试（省略可选字段并收敛 max_tokens）"
                )
                try:
                    body = post(fallback_payload)
                except Exception as e2:
                    raise LLMServiceError(
                        f"LM Studio 请求异常: {e} | 兼容重试仍失败: {e2}"
                    ) from e2
            else:
                raise LLMServiceError(f"LM Studio 请求异常: {e}") from e

        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            raise LLMServiceError(f"LM Studio 返回非 JSON: {body[:500]}")

        choices = obj.get("choices") or []
        if not choices:
            raise LLMServiceError(f"LM Studio 返回无 choices: {obj}")
        msg = choices[0].get("message") or {}
        if not isinstance(msg, dict):
            raise LLMServiceError(f"LM Studio 返回格式异常: {obj}")
        return self._lm_studio_extract_visible_text(msg)

    def call_llm_api(
        self, prompt: str, progress_callback: Optional[Callable] = None
    ) -> str:
        if self.backend == "lm_studio":
            return self._call_lm_studio_api(prompt, progress_callback)
        return self._call_ollama_api(prompt, progress_callback)

    def _call_ollama_api(
        self, prompt: str, progress_callback: Optional[Callable] = None
    ) -> str:
        try:
            import ollama

            messages = self._build_chat_messages(prompt)
            options = {"num_thread": self.num_thread, "temperature": 0.1}

            logger.info(
                f"使用 ollama Python 包: {self.model} (Threads: {self.num_thread})"
            )

            response = ollama.chat(model=self.model, messages=messages, options=options)

            if "message" in response and "content" in response["message"]:
                raw = response["message"]["content"]
                return self._strip_gemma_reasoning_prefix(
                    raw, self._is_gemma4_family_model()
                )
            if "response" in response:
                return response["response"]
            raise LLMServiceError(f"Ollama 返回格式异常: {response}")

        except ImportError:
            logger.info(f"使用命令行调用 Ollama: {self.model}")
            return self._call_ollama_cli(prompt, progress_callback)
        except Exception as e:
            logger.warning(f"ollama 包失败: {e}，尝试命令行")
            return self._call_ollama_cli(prompt, progress_callback)

    def _call_ollama_cli(
        self, prompt: str, progress_callback: Optional[Callable] = None
    ) -> str:
        try:
            if self._is_gemma4_family_model():
                if self.lm_enable_thinking:
                    prompt = "<|think|>\n" + prompt
                else:
                    prompt = (
                        "Thinking mode is OFF: output only ID|translation lines, "
                        "no chain-of-thought or [Start thinking] sections.\n\n"
                        + prompt
                    )
            cmd = ["ollama", "run", self.model]
            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(self.num_thread)

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=env,
            )

            prompt_with_newline = prompt + "\n"
            stdout, stderr = process.communicate(
                input=prompt_with_newline, timeout=self.timeout
            )

            if process.returncode != 0:
                raise LLMServiceError(
                    f"Ollama 命令失败: {stderr if stderr else '未知错误'}"
                )

            result = stdout.strip()
            if not result:
                raise LLMServiceError("Ollama 返回为空")
            return self._strip_gemma_reasoning_prefix(
                result, self._is_gemma4_family_model()
            )

        except subprocess.TimeoutExpired:
            if "process" in locals():
                process.kill()
            raise LLMServiceError(f"Ollama 超时（{self.timeout} 秒）")
        except Exception as e:
            raise LLMServiceError(f"调用 Ollama 失败: {str(e)}")

    def _is_lm_studio_socket_timeout_error(self, e: BaseException) -> bool:
        if self.backend != "lm_studio":
            return False
        if isinstance(e, TimeoutError):
            return True
        msg = str(e).lower()
        return "timed out" in msg or "read timed out" in msg

    def translate_with_retry(
        self, prompt: str, progress_callback: Optional[Callable] = None
    ) -> str:
        last_error = None
        for attempt in range(self.retry_times):
            try:
                if attempt > 0:
                    logger.info(f"重试翻译（第{attempt + 1}次）...")
                    time.sleep(self.retry_delay)
                return self.call_llm_api(prompt, progress_callback)
            except Exception as e:
                last_error = e
                logger.warning(f"翻译失败（第{attempt + 1}次尝试）: {e}")
                if self._is_lm_studio_socket_timeout_error(e):
                    logger.info(
                        "LM Studio 在等待响应时超时，不再自动重试（可在 config 中将 "
                        "lm_studio.timeout 设为 null 表示不限制等待时间，或增大秒数）"
                    )
                    raise LLMServiceError(
                        f"LM Studio 请求超时: {last_error}"
                    ) from last_error
        raise LLMServiceError(f"翻译失败，已重试{self.retry_times}次: {last_error}")

    def _is_valid_translation(
        self, original: str, translated: str, target_lang: str
    ) -> bool:
        if not translated or not translated.strip():
            return False

        clean_org = original.strip()
        clean_trans = translated.strip()

        if len(clean_org) < 3 or clean_org.isdigit():
            return True

        def _has_cjk(s: str) -> bool:
            return any("\u4e00" <= char <= "\u9fff" for char in s)

        def _has_hangul(s: str) -> bool:
            return any("\uac00" <= char <= "\ud7a3" for char in s)

        def _has_kana(s: str) -> bool:
            for c in s:
                o = ord(c)
                if 0x3040 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF:
                    return True
            return False

        def _has_cyrillic(s: str) -> bool:
            for c in s:
                o = ord(c)
                if 0x0400 <= o <= 0x04FF:
                    return True
            return False

        def _has_arabic_script(s: str) -> bool:
            for c in s:
                o = ord(c)
                if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F:
                    return True
            return False

        if target_lang in ("zh-Hans", "zh-Hant"):
            org_has_cjk = _has_cjk(clean_org)
            trans_has_cjk = _has_cjk(clean_trans)
            if trans_has_cjk:
                pass
            elif org_has_cjk:
                return False
            else:
                if clean_org.lower() == clean_trans.lower():
                    if clean_org.istitle() and len(clean_org) < 25:
                        return True
                    if " " in clean_org or len(clean_org) > 15:
                        return False
            return True

        if target_lang == "ko":
            if _has_hangul(clean_trans):
                return True
            if len(clean_org) > 12 and clean_org.lower() == clean_trans.lower():
                return False
            if not _has_cjk(clean_org) and len(clean_org) > 10 and not _has_hangul(
                clean_trans
            ):
                return False
            return True

        if target_lang == "ja":
            if _has_cjk(clean_trans) or _has_kana(clean_trans):
                return True
            if len(clean_org) > 12 and clean_org.lower() == clean_trans.lower():
                return False
            if not _has_cjk(clean_org) and len(clean_org) > 10 and not (
                _has_kana(clean_trans) or _has_cjk(clean_trans)
            ):
                return False
            return True

        if target_lang == "ru":
            if _has_cyrillic(clean_trans):
                return True
            if len(clean_org) > 12 and clean_org.lower() == clean_trans.lower():
                return False
            return True

        if target_lang in ("ar", "fa", "ur"):
            if _has_arabic_script(clean_trans):
                return True
            if len(clean_org) > 12 and clean_org.lower() == clean_trans.lower():
                return False
            return True

        if target_lang == "he":

            def _has_hebrew(s: str) -> bool:
                for c in s:
                    o = ord(c)
                    if 0x0590 <= o <= 0x05FF:
                        return True
                return False

            if _has_hebrew(clean_trans):
                return True
            if len(clean_org) > 12 and clean_org.lower() == clean_trans.lower():
                return False
            return True

        if target_lang == "hi":

            def _has_devanagari(s: str) -> bool:
                for c in s:
                    o = ord(c)
                    if 0x0900 <= o <= 0x097F:
                        return True
                return False

            if _has_devanagari(clean_trans):
                return True
            if len(clean_org) > 12 and clean_org.lower() == clean_trans.lower():
                return False
            return True

        if len(clean_org) > 12 and clean_org.lower() == clean_trans.lower():
            return False
        return True

    def _fill_missing_batch_translations(
        self,
        missing_keys: List[str],
        batch_dict: Dict[str, Dict],
        valid_translations: Dict[str, str],
        text_processor,
        src_en: str,
        tgt_en: str,
        domain_instruction: str,
        glossary_instruction: str,
        progress_callback: Optional[Callable],
        target_lang: str,
        max_rounds: int = 3,
    ) -> Dict[str, str]:
        """仅对模型确实未返回的条目进行补译请求。"""
        out = dict(valid_translations)
        remaining = [k for k in missing_keys if k not in out]
        if not remaining:
            return out

        for round_idx in range(max_rounds):
            fill_dict = {k: batch_dict[k] for k in remaining}
            formatted = text_processor.format_text_for_translation(fill_dict)
            supplement = (
                f"\n\nSUPPLEMENT (mandatory): These {len(remaining)} input line(s) were missing "
                f"from a previous batch reply. Output EXACTLY {len(remaining)} lines, in the same "
                "order as below, each line ID|translation only. Include the very LAST line. "
                "Do not merge adjacent lines; translate only the text after each line's colon."
            )
            prompt = (
                text_processor.create_translation_prompt(
                    formatted,
                    src_en,
                    tgt_en,
                    domain_instruction=domain_instruction,
                    glossary_instruction=glossary_instruction,
                )
                + supplement
            )
            try:
                result_text = self.translate_with_retry(prompt, progress_callback)
                print("--- 补译缺失条目 ---")
                print(formatted)
                print(f"--- Fill round {round_idx + 1} ---")
                print(result_text)
                parsed = text_processor.parse_translation_result(result_text)
                matched = text_processor.smart_match_translations(parsed, remaining)
                for key, trans_text in matched.items():
                    if key not in remaining:
                        continue
                    # 只要模型返回了内容就接受（专有名词/文件名等与原文一致属于正常情况）
                    if trans_text and trans_text.strip():
                        out[key] = trans_text
                new_remaining = [k for k in missing_keys if k not in out]
                logger.info(
                    f"补译第 {round_idx + 1} 轮: 解析 {len(parsed)} 条, "
                    f"匹配 {len(matched)} 条, 仍缺 {len(new_remaining)}/{len(missing_keys)}"
                )
                remaining = new_remaining
                if not remaining:
                    break
            except Exception as e:
                logger.error(f"补译失败: {e}")
                break

        if remaining:
            logger.warning(
                f"补译后仍缺失 {len(remaining)} 条: {remaining[:8]}"
                f"{'...' if len(remaining) > 8 else ''}"
            )
        return out

    def translate_batch(
        self,
        numbered_texts: Dict[str, Dict],
        progress_callback: Optional[Callable] = None,
        *,
        source_lang: str = "en",
        target_lang: str = "zh-Hans",
        industry_preset: str = "general",
        industry_custom_text: str = "",
        glossary_file_path: str = "",
    ) -> Dict[str, str]:
        from core.text_processor import TextProcessor

        text_processor = TextProcessor()
        all_translations = {}

        src_en = english_name_for_code(source_lang)
        tgt_en = english_name_for_code(target_lang)
        domain_instruction = build_domain_instruction_english(
            industry_preset, industry_custom_text
        )
        glossary_instruction = ""
        if glossary_file_path and str(glossary_file_path).strip():
            glossary_instruction = load_glossary_file(str(glossary_file_path).strip())
            if glossary_instruction:
                logger.info("已加载个人专业词汇表并写入提示词")
            else:
                logger.info("专业词汇表路径已设置但无有效条目，跳过词汇表块")

        items = list(numbered_texts.items())
        total_items = len(items)
        try:
            batch_size = int(self.config.get("translation.batch_size", 20) or 20)
        except (TypeError, ValueError):
            batch_size = 20
        batch_size = max(4, min(batch_size, 48))

        for batch_start in range(0, total_items, batch_size):
            batch_end = min(batch_start + batch_size, total_items)
            batch_items = items[batch_start:batch_end]
            batch_dict = dict(batch_items)
            batch_keys_order = [item[0] for item in batch_items]

            formatted_text = text_processor.format_text_for_translation(batch_dict)
            n_lines = len(batch_dict)
            current_prompt = text_processor.create_translation_prompt(
                formatted_text,
                src_en,
                tgt_en,
                domain_instruction=domain_instruction,
                glossary_instruction=glossary_instruction,
            )
            current_prompt += (
                f"\n\nLINE_COUNT_CHECK: The INPUT block above contains exactly "
                f"{n_lines} non-empty lines. "
                f"Your OUTPUT must contain exactly {n_lines} lines, each one "
                f"ID|translation. "
                f"Do not merge lines, do not skip lines, do not stop early. "
                f"The last line of your output must translate the LAST line of "
                f"the input (same ID as that line)."
            )

            if progress_callback:
                progress_callback(
                    batch_start,
                    total_items,
                    f"正在翻译第 {batch_start + 1}-{batch_end} 个段落...",
                )

            logger.info(
                f"翻译批次 {batch_start // batch_size + 1}，"
                f"包含 {len(batch_dict)} 个段落"
            )

            # ----------------------------------------------------------
            # 整批重试仅在质量极低（success_rate <= 0.1）时触发；
            # 质量合格但有缺失条目时直接跳出循环走补译流程。
            # ----------------------------------------------------------
            max_quality_retries = 3
            best_matched = {}

            for attempt in range(max_quality_retries):
                try:
                    result_text = self.translate_with_retry(
                        current_prompt, progress_callback
                    )

                    print("--- 原始数据 ---")
                    print(formatted_text)
                    print(f"--- Batch Output Attempt {attempt + 1} ---")
                    print(result_text)

                    parsed_translations = (
                        text_processor.parse_translation_result(result_text)
                    )

                    logger.info(
                        f"解析结果: {len(parsed_translations)} 个翻译"
                    )
                    logger.debug(
                        f"解析编号示例: "
                        f"{list(parsed_translations.keys())[:3]}"
                    )
                    logger.debug(
                        f"期望编号示例: {batch_keys_order[:3]}"
                    )

                    matched_translations = (
                        text_processor.smart_match_translations(
                            parsed_translations, batch_keys_order
                        )
                    )

                    logger.info(
                        f"智能匹配: {len(matched_translations)}/"
                        f"{len(batch_keys_order)} 个成功匹配"
                    )

                    # 校验仅用于质量统计，不影响重试/补译决策
                    valid_count = 0
                    for key, trans_text in matched_translations.items():
                        original_item = batch_dict.get(key)
                        original_text = (
                            original_item["text"] if original_item else ""
                        )
                        if self._is_valid_translation(
                            original_text, trans_text, target_lang
                        ):
                            valid_count += 1

                    n_batch = len(batch_dict)
                    success_rate = (
                        valid_count / n_batch if n_batch > 0 else 0
                    )
                    fully_matched = len(matched_translations) >= n_batch

                    # ---------- 质量合格：保留所有已匹配结果并跳出 ----------
                    if success_rate > 0.1:
                        best_matched = dict(matched_translations)

                        if fully_matched:
                            if valid_count < n_batch:
                                logger.info(
                                    f"✓ 批次翻译完成 "
                                    f"(匹配 {len(matched_translations)}"
                                    f"/{n_batch}, "
                                    f"其中 {n_batch - valid_count} 条"
                                    f"与原文一致，"
                                    f"视为专有名词/文件名等保留原文)"
                                )
                            else:
                                logger.info(
                                    f"✓ 批次翻译质量合格 "
                                    f"(有效率 {success_rate:.0%})"
                                )
                        else:
                            missing_count = (
                                n_batch - len(matched_translations)
                            )
                            logger.info(
                                f"批次质量合格 "
                                f"(匹配 {len(matched_translations)}"
                                f"/{n_batch}, "
                                f"缺失 {missing_count} 条)，"
                                f"将直接补译缺失条目"
                            )
                        # 无论是否完整，质量合格即跳出，缺失由后续补译处理
                        break

                    # ---------- 质量极低：整批重试 ----------
                    logger.warning(
                        f"批次翻译质量低 "
                        f"(有效率 {success_rate:.0%})，"
                        f"正在重试 "
                        f"({attempt + 1}/{max_quality_retries})..."
                    )
                    current_prompt += (
                        f"\n\nSYSTEM WARNING: Previous output failed "
                        f"validation. "
                        f"You MUST output each line as ID|translation "
                        f"with the text entirely in {tgt_en}. "
                        f"Do not echo the source language; translate "
                        f"every segment."
                    )

                except Exception as e:
                    logger.error(f"批次逻辑处理异常: {e}")
                    if attempt == max_quality_retries - 1:
                        logger.error("已达到最大重试次数。")

            # ----------------------------------------------------------
            # 仅对模型确实未返回的条目进行补译
            # ----------------------------------------------------------
            missing_keys = [
                k for k in batch_keys_order if k not in best_matched
            ]
            if missing_keys:
                best_matched = self._fill_missing_batch_translations(
                    missing_keys,
                    batch_dict,
                    best_matched,
                    text_processor,
                    src_en,
                    tgt_en,
                    domain_instruction,
                    glossary_instruction,
                    progress_callback,
                    target_lang,
                )

            all_translations.update(best_matched)
            logger.info(
                f"批次完成，累计获得 {len(all_translations)} 个有效翻译"
            )

        return all_translations


# 兼容旧名称
OllamaTranslator = LLMTranslator
