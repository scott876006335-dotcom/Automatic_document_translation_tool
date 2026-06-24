"""GUI主窗口 - 智能平滑进度条版"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import time
from pathlib import Path

# 假设这些模块在你的项目中已正确导入
from core.file_parser import create_parser
from core.translator import LLMTranslator, normalize_lm_studio_base_url
from core.file_generator import create_generator, compute_translation_output_path
from utils.config import Config
from utils.logger import setup_logger
from utils.languages import LANGUAGE_CHOICES
from utils.industry import (
    INDUSTRY_CHOICES,
    industry_id_from_label,
    label_for_industry_id,
)
from utils.converter import FormatConverter  # 确保创建了该文件

logger = setup_logger()

class TranslationWindow:
    """翻译工具主窗口"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("文件自动翻译工具 by XuYuan")
        self.root.geometry("1150x1050")
        self.root.minsize(940, 640)
        
        self.config = Config()
        self.translator = None
        self.is_translating = False
        self.stop_event = threading.Event()
        
        # 变量初始化
        self.single_file_path_var = tk.StringVar()
        self.batch_folder_path_var = tk.StringVar()
        _lm_raw = self.config.get("lm_studio.base_url", "http://127.0.0.1:1234/v1")
        _lm_disp = _lm_raw.rstrip("/")
        if _lm_disp.endswith("/v1"):
            _lm_disp = _lm_disp[:-3]
        self.lm_base_url_var = tk.StringVar(value=_lm_disp or "http://127.0.0.1:1234")
        self.backend_var = tk.StringVar(value=self.config.get("llm.backend", "lm_studio"))
        self.model_var = tk.StringVar(
            value=self.config.get("llm.model")
            or self.config.get("ollama.model", "qwen3.5-35b-a3b")
        )
        self.progress_var = tk.DoubleVar()
        self.industry_custom_var = tk.StringVar(
            value=self.config.get("translation.industry_custom", "") or ""
        )
        self.glossary_path_var = tk.StringVar(
            value=self.config.get("translation.glossary_file", "") or ""
        )
        _bs = self.config.get("translation.batch_size", 20)
        try:
            _bs = int(_bs)
        except (TypeError, ValueError):
            _bs = 20
        _bs = max(4, min(_bs, 48))
        self.batch_size_var = tk.StringVar(value=str(_bs))
        self.thinking_var = tk.BooleanVar(
            value=bool(self.config.get("lm_studio.enable_thinking", False))
        )

        self.file_types_vars = {
            'word': tk.BooleanVar(value=True),
            'ppt': tk.BooleanVar(value=True),
            'excel': tk.BooleanVar(value=True),
            'pdf': tk.BooleanVar(value=False)
        }
        
        self.setup_styles()
        self.setup_ui()
    
    def setup_styles(self):
        """配置界面样式"""
        style = ttk.Style()
        default_font = ("Microsoft YaHei UI", 14)
        style.configure(".", font=default_font)
        style.configure("TLabelframe.Label", font=("Microsoft YaHei UI", 14, "bold"), foreground="#333333")
        style.configure("Big.TButton", font=("Microsoft YaHei UI", 15), padding=(20, 8))
        style.configure("Bold.TLabel", font=("Microsoft YaHei UI", 14, "bold"))
        style.configure("Small.TLabel", font=("Microsoft YaHei UI", 12), foreground="gray")
        style.configure("TNotebook.Tab", font=("Microsoft YaHei UI", 14), padding=(10, 5))
        # 进度条加粗
        style.configure("TProgressbar", thickness=20)

    def setup_ui(self):
        """设置用户界面布局"""
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        # 纵向拉伸时扩展运行日志区域；为中间选项卡保留最小高度，避免批量页底部按钮被裁切
        main_frame.rowconfigure(1, weight=0, minsize=360)
        main_frame.rowconfigure(2, weight=1)

        # === 1. 顶部配置 ===
        config_frame = ttk.LabelFrame(main_frame, text=" 全局参数设置 ", padding="10")
        config_frame.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        
        f0 = ttk.Frame(config_frame)
        f0.pack(fill=tk.X, pady=5)
        ttk.Label(f0, text="推理后端:", style="Bold.TLabel", width=12).pack(side=tk.LEFT)
        ttk.Radiobutton(
            f0, text="LM Studio", variable=self.backend_var, value="lm_studio"
        ).pack(side=tk.LEFT, padx=(5, 15))
        ttk.Radiobutton(f0, text="Ollama", variable=self.backend_var, value="ollama").pack(
            side=tk.LEFT, padx=5
        )

        self._frame_lm_url = ttk.Frame(config_frame)
        ttk.Label(self._frame_lm_url, text="LM Studio 地址:", style="Bold.TLabel", width=12).pack(
            side=tk.LEFT
        )
        ttk.Entry(
            self._frame_lm_url, textvariable=self.lm_base_url_var, width=42, font=("Consolas", 14)
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(
            self._frame_lm_url, text="(OpenAI 兼容 API，一般端口 1234)", style="Small.TLabel"
        ).pack(side=tk.LEFT, padx=5)

        def _sync_lm_url_visibility(*_):
            if self.backend_var.get() == "lm_studio":
                self._frame_lm_url.pack(fill=tk.X, pady=5, after=f0)
            else:
                self._frame_lm_url.pack_forget()

        self.backend_var.trace_add("write", _sync_lm_url_visibility)
        _sync_lm_url_visibility()

        f1 = ttk.Frame(config_frame)
        f1.pack(fill=tk.X, pady=5)
        ttk.Label(f1, text="模型名称:", style="Bold.TLabel", width=12).pack(side=tk.LEFT)
        ttk.Entry(f1, textvariable=self.model_var, width=30, font=("Consolas", 14)).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Label(
            f1, text="(与 LM Studio / Ollama 中加载的名称一致)", style="Small.TLabel"
        ).pack(side=tk.LEFT, padx=10)
        
        self._lang_code_by_label = {row[1]: row[0] for row in LANGUAGE_CHOICES}
        self._lang_labels = [row[1] for row in LANGUAGE_CHOICES]

        def _pick_label_for_code(code: str, fallback_label: str) -> str:
            for row in LANGUAGE_CHOICES:
                if row[0] == code:
                    return row[1]
            return fallback_label

        _src_code = self.config.get("output.source_lang", "en")
        _tgt_code = self.config.get("output.target_lang", "zh-Hans")
        _layout = self.config.get("output.layout")
        if not _layout:
            _layout = self.config.get("output.default_format", "bilingual")
        if _layout == "chinese_only":
            _layout = "target_only"

        f2 = ttk.Frame(config_frame)
        f2.pack(fill=tk.X, pady=5)
        ttk.Label(f2, text="源语言:", style="Bold.TLabel", width=10).pack(side=tk.LEFT)
        self.source_lang_combo = ttk.Combobox(
            f2,
            values=self._lang_labels,
            width=22,
            state="readonly",
            font=("Microsoft YaHei UI", 13),
        )
        self.source_lang_combo.pack(side=tk.LEFT, padx=(0, 18))
        self.source_lang_combo.set(_pick_label_for_code(_src_code, self._lang_labels[0]))

        ttk.Label(f2, text="目标语言:", style="Bold.TLabel", width=10).pack(side=tk.LEFT)
        self.target_lang_combo = ttk.Combobox(
            f2,
            values=self._lang_labels,
            width=22,
            state="readonly",
            font=("Microsoft YaHei UI", 13),
        )
        self.target_lang_combo.pack(side=tk.LEFT, padx=(0, 18))
        self.target_lang_combo.set(_pick_label_for_code(_tgt_code, self._lang_labels[0]))

        ttk.Label(f2, text="输出版式:", style="Bold.TLabel", width=10).pack(side=tk.LEFT)
        self.output_layout_combo = ttk.Combobox(
            f2,
            values=("原文与译文对照", "仅译文"),
            width=16,
            state="readonly",
            font=("Microsoft YaHei UI", 13),
        )
        self.output_layout_combo.pack(side=tk.LEFT, padx=(0, 8))
        self.output_layout_combo.set(
            "仅译文" if _layout in ("target_only", "chinese_only") else "原文与译文对照"
        )

        _ind_labels = [row[1] for row in INDUSTRY_CHOICES]
        _ind_id = self.config.get("translation.industry_preset", "general") or "general"

        f_ind = ttk.Frame(config_frame)
        f_ind.pack(fill=tk.X, pady=5)
        ttk.Label(f_ind, text="翻译行业:", style="Bold.TLabel", width=10).pack(side=tk.LEFT)
        self.industry_combo = ttk.Combobox(
            f_ind,
            values=_ind_labels,
            width=20,
            state="readonly",
            font=("Microsoft YaHei UI", 13),
        )
        self.industry_combo.pack(side=tk.LEFT, padx=(0, 12))
        self.industry_combo.set(label_for_industry_id(_ind_id))
        ttk.Label(
            f_ind,
            text="(术语与风格倾向；选「自定义」时填写下一行)",
            style="Small.TLabel",
        ).pack(side=tk.LEFT, padx=(4, 0))

        f_ind2 = ttk.Frame(config_frame)
        f_ind2.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(f_ind2, text="自定义领域:", style="Bold.TLabel", width=10).pack(
            side=tk.LEFT
        )
        self.entry_industry_custom = ttk.Entry(
            f_ind2,
            textvariable=self.industry_custom_var,
            font=("Microsoft YaHei UI", 13),
        )
        self.entry_industry_custom.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8), ipady=2
        )

        def _sync_industry_custom_state(*_):
            if industry_id_from_label(self.industry_combo.get()) == "custom":
                self.entry_industry_custom.config(state="normal")
            else:
                self.entry_industry_custom.config(state="disabled")

        self.industry_combo.bind("<<ComboboxSelected>>", _sync_industry_custom_state)
        _sync_industry_custom_state()

        f_gl = ttk.Frame(config_frame)
        f_gl.pack(fill=tk.X, pady=5)
        ttk.Label(f_gl, text="专业词汇表:", style="Bold.TLabel", width=10).pack(
            side=tk.LEFT
        )
        ttk.Entry(
            f_gl,
            textvariable=self.glossary_path_var,
            font=("Consolas", 12),
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8), ipady=2)
        ttk.Button(f_gl, text="浏览...", command=self.select_glossary_file).pack(
            side=tk.LEFT
        )
        ttk.Label(
            f_gl,
            text="可选 .txt，每行 原文:译文，详见 usermanual.md",
            style="Small.TLabel",
        ).pack(side=tk.LEFT, padx=(6, 0))

        f_adv = ttk.Frame(config_frame)
        f_adv.pack(fill=tk.X, pady=5)
        ttk.Label(f_adv, text="每批段落数:", style="Bold.TLabel", width=10).pack(
            side=tk.LEFT
        )
        self.batch_spin = tk.Spinbox(
            f_adv,
            from_=4,
            to=48,
            textvariable=self.batch_size_var,
            width=6,
            font=("Consolas", 14),
        )
        self.batch_spin.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(
            f_adv,
            text="（单次请求合并的段落条数，推荐 20 左右）",
            style="Small.TLabel",
        ).pack(side=tk.LEFT, padx=(0, 12))

        self.chk_thinking = ttk.Checkbutton(
            f_adv,
            text="思考模式（更慢，但更准确）",
            variable=self.thinking_var,
        )
        self.chk_thinking.pack(side=tk.LEFT, padx=(8, 0))

        def _sync_thinking_state(*_):
            if self.backend_var.get() == "lm_studio":
                self.chk_thinking.config(state="normal")
            else:
                self.chk_thinking.config(state="disabled")

        self.backend_var.trace_add("write", _sync_thinking_state)
        _sync_thinking_state()

        # === 2. 选项卡 ===
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, sticky="nsew", pady=5)
        
        # Tab 1
        self.tab_single = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(self.tab_single, text="  单文件模式  ")
        self.setup_single_ui(self.tab_single)
        
        # Tab 2
        self.tab_batch = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(self.tab_batch, text="  批量目录模式  ")
        self.setup_batch_ui(self.tab_batch)

        # === 3. 底部 ===
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.rowconfigure(2, weight=1)

        self.progress_bar = ttk.Progressbar(bottom_frame, variable=self.progress_var, maximum=100, style="TProgressbar")
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=5)
        
        self.progress_label = ttk.Label(bottom_frame, text="准备就绪", anchor="center", font=("Microsoft YaHei UI", 13))
        self.progress_label.grid(row=1, column=0, pady=2)

        log_group = ttk.LabelFrame(bottom_frame, text=" 运行日志 ", padding="5")
        log_group.grid(row=2, column=0, sticky="nsew", pady=5)
        log_group.columnconfigure(0, weight=1)
        log_group.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_group, height=8, state=tk.DISABLED, font=("Consolas", 14))
        self.log_text.grid(row=0, column=0, sticky="nsew")

        self.log("程序启动完成。")
        if self.backend_var.get() == "lm_studio":
            self.log(
                "LM Studio：可在上方切换「思考模式」。Gemma 4 会注入 system、"
                "chat_template_kwargs，并在关闭思考时附带 reasoning_budget=0；"
                "若服务端不支持部分字段，请求可能自动回退（详见 readme）。"
            )

    def setup_single_ui(self, parent):
        container = ttk.Frame(parent)
        container.pack(fill=tk.X) 
        
        file_group = ttk.LabelFrame(container, text=" 文件选择 ", padding=(15, 10))
        file_group.pack(fill=tk.X, pady=(0, 20))
        
        f_input = ttk.Frame(file_group)
        f_input.pack(fill=tk.X)
        
        self.entry_single = ttk.Entry(f_input, textvariable=self.single_file_path_var, state="readonly", font=("Microsoft YaHei UI", 14))
        self.entry_single.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10), ipady=3)
        ttk.Button(f_input, text="浏览...", command=self.select_single_file).pack(side=tk.LEFT)
        
        ttk.Label(file_group, text="支持: .docx, .doc, .pptx, .ppt, .xlsx, .xls, .pdf 等", style="Small.TLabel").pack(anchor="w", pady=(8,0))

        btn_frame = ttk.Frame(container)
        btn_frame.pack(pady=10)
        self.btn_start_single = ttk.Button(btn_frame, text="开始翻译", style="Big.TButton", 
                                         command=lambda: self.start_translation(mode='single'))
        self.btn_start_single.pack(side=tk.LEFT, padx=20)
        self.btn_cancel_single = ttk.Button(btn_frame, text="停止任务", style="Big.TButton", 
                                          command=self.cancel_translation, state=tk.DISABLED)
        self.btn_cancel_single.pack(side=tk.LEFT, padx=20)

    def setup_batch_ui(self, parent):
        # 先固定底部按钮区，再自上而下排内容，避免在部分 DPI/环境下贴底被裁切
        action_frame = ttk.Frame(parent)
        action_frame.pack(side=tk.BOTTOM, pady=(12, 4))
        self.btn_start_batch = ttk.Button(action_frame, text="批量翻译", style="Big.TButton",
                                        command=lambda: self.start_translation(mode='batch'))
        self.btn_start_batch.pack(side=tk.LEFT, padx=20)
        self.btn_cancel_batch = ttk.Button(action_frame, text="停止任务", style="Big.TButton",
                                         command=self.cancel_translation, state=tk.DISABLED)
        self.btn_cancel_batch.pack(side=tk.LEFT, padx=20)

        path_group = ttk.LabelFrame(parent, text=" 1. 目标文件夹 ", padding=(15, 10))
        path_group.pack(side=tk.TOP, fill=tk.X, pady=(0, 15))
        
        f_path = ttk.Frame(path_group)
        f_path.pack(fill=tk.X)
        self.entry_batch = ttk.Entry(f_path, textvariable=self.batch_folder_path_var, state="readonly", font=("Microsoft YaHei UI", 14))
        self.entry_batch.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10), ipady=3)
        ttk.Button(f_path, text="浏览...", command=self.select_batch_folder).pack(side=tk.LEFT)

        type_group = ttk.LabelFrame(parent, text=" 2. 筛选文件类型 ", padding=(15, 10))
        type_group.pack(side=tk.TOP, fill=tk.X, pady=(0, 15))
        
        ttk.Checkbutton(type_group, text="Word 文档", variable=self.file_types_vars['word']).grid(row=0, column=0, sticky="w", padx=20, pady=8)
        ttk.Label(type_group, text="(.docx, .docm, .doc)", style="Small.TLabel").grid(row=0, column=1, sticky="w")
        
        ttk.Checkbutton(type_group, text="Excel 表格", variable=self.file_types_vars['excel']).grid(row=0, column=2, sticky="w", padx=20, pady=8)
        ttk.Label(type_group, text="(.xlsx, .xlsm, .xls)", style="Small.TLabel").grid(row=0, column=3, sticky="w")
        
        ttk.Checkbutton(type_group, text="PPT 演示文稿", variable=self.file_types_vars['ppt']).grid(row=1, column=0, sticky="w", padx=20, pady=8)
        ttk.Label(type_group, text="(.pptx, .pptm, .ppt)", style="Small.TLabel").grid(row=1, column=1, sticky="w")
        
        ttk.Checkbutton(type_group, text="PDF 文档", variable=self.file_types_vars['pdf']).grid(row=1, column=2, sticky="w", padx=20, pady=8)
        ttk.Label(type_group, text="(.pdf)", style="Small.TLabel").grid(row=1, column=3, sticky="w")
        
        type_group.columnconfigure(1, weight=1)
        type_group.columnconfigure(3, weight=1)

    # === 工具方法 ===
    def _lang_code_from_ui(self, label: str) -> str:
        return self._lang_code_by_label.get(label, LANGUAGE_CHOICES[0][0])

    def _output_layout_code(self) -> str:
        return (
            "target_only"
            if self.output_layout_combo.get() == "仅译文"
            else "bilingual"
        )

    def _persist_translation_prefs(self):
        # 推理后端、模型、LM Studio 地址（与语言/行业等一并写入 config.json，下次启动自动恢复）
        self.config.set("llm.backend", self.backend_var.get().strip() or "lm_studio")
        _m = self.model_var.get().strip()
        self.config.set("llm.model", _m)
        self.config.set("ollama.model", _m)
        _lm = self.lm_base_url_var.get().strip()
        if _lm:
            self.config.set("lm_studio.base_url", normalize_lm_studio_base_url(_lm))

        self.config.set(
            "output.source_lang", self._lang_code_from_ui(self.source_lang_combo.get())
        )
        self.config.set(
            "output.target_lang", self._lang_code_from_ui(self.target_lang_combo.get())
        )
        self.config.set("output.layout", self._output_layout_code())
        self.config.set(
            "translation.industry_preset", industry_id_from_label(self.industry_combo.get())
        )
        self.config.set(
            "translation.industry_custom", self.industry_custom_var.get().strip()
        )
        self.config.set("translation.glossary_file", self.glossary_path_var.get().strip())
        try:
            bs = int((self.batch_size_var.get() or "").strip())
        except ValueError:
            bs = 20
        bs = max(4, min(bs, 48))
        self.batch_size_var.set(str(bs))
        self.config.set("translation.batch_size", bs)
        self.config.set(
            "lm_studio.enable_thinking", bool(self.thinking_var.get())
        )

    def _validated_batch_size(self):
        """返回 4～48 的整数；无效时弹窗并返回 None。"""
        raw = (self.batch_size_var.get() or "").strip()
        try:
            v = int(raw)
        except ValueError:
            messagebox.showwarning("提示", "每批段落数必须是整数。")
            return None
        if v < 4 or v > 48:
            messagebox.showwarning("提示", "每批段落数应在 4～48 之间。")
            return None
        return v

    def _industry_preset_id(self):
        return industry_id_from_label(self.industry_combo.get())

    def _expected_batch_output_path(self, source_file_path: str) -> str:
        """当前界面设置下，该源文件若翻译成功将写入的路径（与生成器命名一致）。"""
        naming_path = FormatConverter.path_used_for_output_naming(source_file_path)
        return compute_translation_output_path(
            naming_path,
            source_lang=self._lang_code_from_ui(self.source_lang_combo.get()),
            target_lang=self._lang_code_from_ui(self.target_lang_combo.get()),
            output_layout=self._output_layout_code(),
            glossary_file_path=self.glossary_path_var.get().strip(),
            industry_preset=self._industry_preset_id(),
        )

    def select_glossary_file(self):
        path = filedialog.askopenfilename(
            title="选择专业词汇表 (.txt)",
            filetypes=[("文本文件", "*.txt"), ("全部", "*.*")],
        )
        if path:
            self.glossary_path_var.set(os.path.normpath(path))
            self.log(f"已选择词汇表: {os.path.basename(path)}")

    def log(self, message):
        def _log():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        self.root.after(0, _log)
        logger.info(message)

    def update_progress(self, value, message=None):
        def _update():
            # 这里的value是0-100的浮点数
            self.progress_var.set(value)
            if message:
                self.progress_label.config(text=message)
        self.root.after(0, _update)

    def select_single_file(self):
        file_path = filedialog.askopenfilename(title="选择文件", filetypes=[("Docs", "*.docx;*.docm;*.doc;*.pptx;*.pptm;*.ppt;*.xlsx;*.xlsm;*.xls;*.pdf")])
        if file_path:
            self.single_file_path_var.set(file_path)
            self.log(f"已选择: {os.path.basename(file_path)}")

    def select_batch_folder(self):
        folder_path = filedialog.askdirectory(title="选择文件夹")
        if folder_path:
            self.batch_folder_path_var.set(os.path.normpath(folder_path))
            self.log(f"已选择目录: {folder_path}")

    def toggle_buttons(self, translating):
        state_start = tk.DISABLED if translating else tk.NORMAL
        state_stop = tk.NORMAL if translating else tk.DISABLED
        self.btn_start_single.config(state=state_start)
        self.btn_cancel_single.config(state=state_stop)
        self.btn_start_batch.config(state=state_start)
        self.btn_cancel_batch.config(state=state_stop)

    def cancel_translation(self):
        if self.is_translating:
            if messagebox.askyesno("停止", "正在等待当前步骤完成后停止..."):
                self.stop_event.set()
                self.log(">>> 收到停止指令...")

    def check_llm(self):
        self.translator = LLMTranslator(self.config)
        self.translator.backend = self.backend_var.get().strip()
        self.translator.model = self.model_var.get().strip()
        self.translator.lm_studio_base_url = normalize_lm_studio_base_url(
            self.lm_base_url_var.get()
        )
        if not self.translator.check_llm_available():
            if self.translator.backend == "lm_studio":
                messagebox.showerror(
                    "错误",
                    "连接 LM Studio 失败，请确认已启动并在「本地服务器」中开启 API（默认 http://127.0.0.1:1234）。",
                )
            else:
                messagebox.showerror(
                    "错误", "连接 Ollama 失败，请检查服务是否运行且 ollama 命令可用。"
                )
            return False
        return True

    def start_translation(self, mode='single'):
        if self.is_translating: return

        if mode == 'single':
            path = self.single_file_path_var.get()
            if not path or not os.path.exists(path):
                messagebox.showwarning("提示", "请先选择文件")
                return
            target = self.run_single_task
            args = (path,)
        else:
            path = self.batch_folder_path_var.get()
            if not path or not os.path.exists(path):
                messagebox.showwarning("提示", "请先选择文件夹")
                return
            target = self.run_batch_task
            args = ()

        src = self._lang_code_from_ui(self.source_lang_combo.get())
        tgt = self._lang_code_from_ui(self.target_lang_combo.get())
        if src == tgt:
            messagebox.showwarning("提示", "源语言与目标语言不能相同，请修改后重试。")
            return

        bs = self._validated_batch_size()
        if bs is None:
            return
        self.batch_size_var.set(str(bs))

        self._persist_translation_prefs()

        if not self.check_llm():
            return

        self.is_translating = True
        self.stop_event.clear()
        self.toggle_buttons(True)
        threading.Thread(target=target, args=args, daemon=True).start()

    # === 核心逻辑优化：平滑进度条 ===

    def process_one_file(self, file_path, file_index_0_based, total_files):
        original_file_name = os.path.basename(file_path)
        file_width = 100.0 / total_files
        base_progress = file_index_0_based * file_width
        
        converted_path = None
        is_temp = False
        
        try:
            if self.stop_event.is_set(): return False
            
            # === 0. 格式预处理 (老格式转新格式) ===
            self.log(f"[{file_index_0_based + 1}/{total_files}] 正在预处理: {original_file_name}")
            
            # 调用转换器
            actual_process_path, is_temp = FormatConverter.convert_to_modern(file_path)
            
            if not actual_process_path:
                self.log(f"❌ 跳过: 无法转换格式 {original_file_name}")
                return False
                
            if is_temp:
                self.log(f"   已转换旧格式 -> {os.path.basename(actual_process_path)}")

            # === 1. 解析阶段 ===
            # 使用转换后的路径进行解析
            self.update_progress(base_progress + (file_width * 0.05), f"正在解析...")
            
            # create_parser 传入 actual_process_path
            parser = create_parser(actual_process_path)
            if not parser:
                self.log(f"❌ 跳过: 格式不支持")
                return False
                
            numbered_texts = parser.parse()
            if not numbered_texts:
                self.log(f"⚠️ 跳过: 无可译文本")
                return True 

            # === 2. 翻译阶段 ===
            # ... (保持原代码不变) ...
            total_segments = len(numbered_texts)
            self.log(f"   共 {total_segments} 段，开始翻译...")
            
            def progress_cb(cur, tot, msg):
                ratio = cur / tot
                increment = ratio * (file_width * 0.90)
                global_val = base_progress + (file_width * 0.05) + increment
                self.update_progress(global_val, f"翻译中 ({cur}/{tot})")
            
            translations = self.translator.translate_batch(
                numbered_texts,
                progress_callback=progress_cb,
                source_lang=self._lang_code_from_ui(self.source_lang_combo.get()),
                target_lang=self._lang_code_from_ui(self.target_lang_combo.get()),
                industry_preset=self._industry_preset_id(),
                industry_custom_text=self.industry_custom_var.get(),
                glossary_file_path=self.glossary_path_var.get().strip(),
            )
            
            if self.stop_event.is_set(): return False

            # === 3. 生成阶段 ===
            self.update_progress(base_progress + (file_width * 0.95), f"正在生成文件...")
            
            # 注意：generator 也要基于 actual_process_path 生成，这样生成的是新格式文件
            generator = create_generator(
                actual_process_path,
                translations,
                numbered_texts,
                source_lang=self._lang_code_from_ui(self.source_lang_combo.get()),
                target_lang=self._lang_code_from_ui(self.target_lang_combo.get()),
                output_layout=self._output_layout_code(),
                glossary_file_path=self.glossary_path_var.get().strip(),
                industry_preset=self._industry_preset_id(),
            )
            output_path = generator.generate(self._output_layout_code())
            
            # 如果原文件是旧格式，output_path 会是 .docx/.xlsx/.pptx
            # 我们可以重命名它，或者告诉用户已保存为新格式
            
            # === 4. 清理 ===
            if is_temp and os.path.exists(actual_process_path):
                try:
                    os.remove(actual_process_path) # 删除中间临时文件
                except:
                    pass
            
            self.update_progress(base_progress + file_width, f"完成: {original_file_name}")
            self.log(f"✅ 保存至: {os.path.basename(output_path)}")
            return True

        except Exception as e:
            self.log(f"❌ 错误: {str(e)}")
            logger.error(f"Error: {file_path} -> {e}")
            # 发生错误也要尝试清理临时文件
            if is_temp and converted_path and os.path.exists(converted_path):
                try: os.remove(converted_path)
                except: pass
            return False

    def run_single_task(self, file_path):
        try:
            # 单文件模式：index=0, total=1
            # 这样 base=0, width=100，逻辑完美兼容
            self.process_one_file(file_path, 0, 1)
            self.update_progress(100, "完成")
            if not self.stop_event.is_set():
                messagebox.showinfo("成功", "翻译完成")
        finally:
            self.is_translating = False
            self.toggle_buttons(False)

    def run_batch_task(self):
        try:
            self.log("扫描文件中...")
            target_exts = []
            if self.file_types_vars['word'].get(): target_exts.extend(['.docx', '.docm', '.doc'])
            if self.file_types_vars['ppt'].get(): target_exts.extend(['.pptx', '.pptm', '.ppt'])
            if self.file_types_vars['excel'].get(): target_exts.extend(['.xlsx', '.xlsm', '.xls'])
            if self.file_types_vars['pdf'].get(): target_exts.append('.pdf')
            target_exts = set(ext.lower() for ext in target_exts)
            
            if not target_exts:
                self.log("请选择文件类型")
                return

            marked_xuyuan = []
            sources_raw = []
            for root, dirs, files in os.walk(self.batch_folder_path_var.get()):
                for file in files:
                    if file.startswith('~$'):
                        continue
                    ext = os.path.splitext(file)[1].lower()
                    if ext not in target_exts:
                        continue
                    full = os.path.join(root, file)
                    stem = Path(file).stem
                    if stem.endswith("_XuYuan"):
                        marked_xuyuan.append(full)
                    else:
                        sources_raw.append(full)

            pending = []
            skipped_done = []
            for fp in sources_raw:
                exp = self._expected_batch_output_path(fp)
                if os.path.isfile(exp):
                    skipped_done.append(fp)
                else:
                    pending.append(fp)

            n_marked = len(marked_xuyuan)
            n_skip = len(skipped_done)
            self.log(
                f"扫描：已署名输出 *_XuYuan 共 {n_marked} 个；"
                f"待译候选源文件 {len(sources_raw)} 个；"
                f"按当前设置译稿已存在跳过 {n_skip} 个，待译 {len(pending)} 个。"
            )
            if n_skip and skipped_done:
                show = min(5, len(skipped_done))
                for p in skipped_done[:show]:
                    self.log(f"   跳过(已有译稿): {os.path.basename(p)}")
                if len(skipped_done) > show:
                    self.log(f"   … 另有 {len(skipped_done) - show} 个，略。")

            all_files = pending
            total = len(all_files)
            if total == 0:
                self.log("没有需要翻译的文件（已全部跳过或目录为空）")
                return

            self.log(f"开始处理 {total} 个文件")
            success, fail = 0, 0
            
            for i, fp in enumerate(all_files):
                if self.stop_event.is_set(): break
                
                # 注意：这里 i 是从 0 开始的，正好传入 process_one_file
                if self.process_one_file(fp, i, total): 
                    success += 1
                else: 
                    fail += 1
                
                # 稍微停顿，让UI渲染最后的状态
                time.sleep(0.2)

            self.update_progress(100, "批量结束")
            messagebox.showinfo("完成", f"成功: {success}\n失败/跳过: {fail}")

        except Exception as e:
            self.log(f"批量错误: {e}")
        finally:
            self.is_translating = False
            self.toggle_buttons(False)

def main():
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except: pass
    app = TranslationWindow(root)
    root.mainloop()

if __name__ == "__main__":
    main()
