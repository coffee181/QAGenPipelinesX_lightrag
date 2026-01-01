"""文件级进度管理（progress.jsonl）。

以文件元数据（修改时间 + 大小）为第一层校验，不使用 MD5。
每条记录使用 JSONL 存储，键为项目根路径下的相对路径。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

from ..utils.config import ConfigManager
from ..utils.file_utils import FileUtils


class ProgressManager:
    """基于 JSONL 的文件进度管理器。"""

    DEFAULT_STAGES = ("preprocess", "qa_gen", "vectorization")
    OPERATION_STAGE_MAP = {
        "pdf_processing": "preprocess",
        "pdf_list_processing": "preprocess",
        "question_generation": "qa_gen",
        "question_doc_generation": "qa_gen",
        "answer_generation": "vectorization",
        "batch_answer_generation": "vectorization",
        "document_insertion": "vectorization",
    }

    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        progress_file: Optional[Path | str] = None,
        project_root: Optional[Path | str] = None,
    ) -> None:
        self.project_root = self._resolve_project_root(project_root)
        progress_path = progress_file
        if progress_path is None and config is not None:
            progress_path = config.get("progress.progress_file", "progress.jsonl")
        self.progress_file = self._resolve_storage_path(progress_path or "progress.jsonl")

        self.records: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}

        self._load()
        logger.info("ProgressManager initialized at %s", self.progress_file)

    # ------------------------------------------------------------------#
    # 核心 JSONL 读写
    # ------------------------------------------------------------------#
    def _resolve_project_root(self, root: Optional[Path | str]) -> Path:
        if root:
            return Path(root).resolve()
        return Path(__file__).resolve().parents[2]

    def _resolve_storage_path(self, path_like: Path | str) -> Path:
        candidate = Path(path_like)
        if not candidate.is_absolute():
            candidate = (self.project_root / candidate).resolve()
        return candidate

    def _load(self) -> None:
        if not self.progress_file.exists():
            return

        try:
            with self.progress_file.open("r", encoding="utf-8") as fp:
                for line in fp:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("跳过无法解析的 progress 行: %s", line.strip())
                        continue
                    normalized = self._normalize_record(record)
                    self.records[normalized["file_path"]] = normalized
        except Exception as exc:  # noqa: BLE001
            logger.error("读取 progress.jsonl 失败: %s", exc)

    def _persist(self) -> None:
        tmp_path = self.progress_file.with_suffix(self.progress_file.suffix + ".tmp")
        FileUtils.ensure_directory(tmp_path.parent)

        try:
            with tmp_path.open("w", encoding="utf-8") as fp:
                for _, record in sorted(self.records.items(), key=lambda item: item[0]):
                    fp.write(json.dumps(record, ensure_ascii=False))
                    fp.write("\n")
            tmp_path.replace(self.progress_file)
        except Exception as exc:  # noqa: BLE001
            logger.error("写入 progress.jsonl 失败: %s", exc)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------#
    # 记录与元数据
    # ------------------------------------------------------------------#
    def _default_stage_states(self) -> Dict[str, Dict[str, Any]]:
        return {stage: {"status": "pending"} for stage in self.DEFAULT_STAGES}

    def _normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        stages = record.get("stages") or {}
        for stage in self.DEFAULT_STAGES:
            stages.setdefault(stage, {"status": "pending"})

        return {
            "file_path": record["file_path"],
            "file_size": int(record.get("file_size", 0)),
            "last_modified": float(record.get("last_modified", 0.0)),
            "stages": stages,
        }

    def _build_record(self, rel_path: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "file_path": rel_path,
            "file_size": meta["file_size"],
            "last_modified": meta["last_modified"],
            "stages": self._default_stage_states(),
        }

    def _relative_path(self, file_path: Path) -> str:
        try:
            return str(file_path.resolve().relative_to(self.project_root))
        except ValueError:
            return os.path.relpath(file_path.resolve(), self.project_root)

    def _get_metadata(self, file_path: Path) -> Dict[str, Any]:
        stat = file_path.stat()
        return {"file_size": stat.st_size, "last_modified": stat.st_mtime}

    def _metadata_changed(self, record: Dict[str, Any], meta: Dict[str, Any]) -> bool:
        return (
            record.get("file_size") != meta["file_size"]
            or record.get("last_modified") != meta["last_modified"]
        )

    # ------------------------------------------------------------------#
    # 对外接口：跳过 / 更新
    # ------------------------------------------------------------------#
    def should_skip(self, file_path: Path | str, stage_name: str) -> bool:
        """返回是否跳过给定阶段；必要时会创建或重置记录。"""
        if not stage_name:
            return False

        path_obj = Path(file_path).expanduser().resolve()
        if not path_obj.exists():
            raise FileNotFoundError(f"文件不存在: {path_obj}")

        stage = stage_name.lower()
        rel_path = self._relative_path(path_obj)
        meta = self._get_metadata(path_obj)

        record = self.records.get(rel_path)
        if record is None:
            record = self._build_record(rel_path, meta)
            self.records[rel_path] = record
            self._persist()
            return False

        if self._metadata_changed(record, meta):
            record["file_size"] = meta["file_size"]
            record["last_modified"] = meta["last_modified"]
            record["stages"] = self._default_stage_states()
            self.records[rel_path] = record
            self._persist()
            return False

        stage_state = record["stages"].setdefault(stage, {"status": "pending"})
        return stage_state.get("status") == "done"

    def update_status(self, file_path: Path | str, stage_name: str, status: str) -> None:
        """更新阶段状态并持久化到 JSONL。"""
        if not stage_name:
            return

        path_obj = Path(file_path).expanduser().resolve()
        stage = stage_name.lower()
        rel_path = self._relative_path(path_obj)
        meta = self._get_metadata(path_obj)

        record = self.records.get(rel_path)
        if record is None:
            record = self._build_record(rel_path, meta)
        else:
            record["file_size"] = meta["file_size"]
            record["last_modified"] = meta["last_modified"]

        stage_state = record["stages"].setdefault(stage, {"status": "pending"})
        stage_state["status"] = status
        stage_state["timestamp"] = time.time()
        record["stages"][stage] = stage_state

        self.records[rel_path] = record
        self._persist()

    # ------------------------------------------------------------------#
    # 向后兼容的会话接口（仅内存，用于现有服务调用）
    # ------------------------------------------------------------------#
    def _stage_for_operation(self, operation_type: Optional[str]) -> Optional[str]:
        if not operation_type:
            return None
        return self.OPERATION_STAGE_MAP.get(operation_type)

    def create_session(
        self,
        session_id: str,
        operation_type: str,
        total_items: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.sessions[session_id] = {
            "session_id": session_id,
            "operation_type": operation_type,
            "total_items": total_items,
            "completed_items": 0,
            "failed_items": 0,
            "processed_files": [],
            "failed_files": [],
            "status": "running",
            "metadata": metadata or {},
            "start_time": time.time(),
            "last_update": time.time(),
        }

    def start_session(
        self,
        session_id: str,
        total_items: int,
        operation_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.create_session(session_id, operation_type, total_items, metadata)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话元数据；无则返回 None。"""
        return self.sessions.get(session_id)

    def add_error(self, session_id: str, error_message: str) -> None:
        """
        记录会话中的错误信息，便于调用方统计失败项。
        若会话不存在则先以兼容模式创建。
        """
        session = self.sessions.get(session_id)
        if session is None:
            logger.debug("Session %s 不存在，自动创建用于记录错误（兼容模式）。", session_id)
            self.create_session(session_id, "unknown", 0, {})
            session = self.sessions[session_id]

        session["failed_items"] += 1
        session["failed_files"].append(
            {
                "error": error_message,
                "timestamp": time.time(),
            }
        )
        session["last_update"] = time.time()

    def update_session_progress(
        self,
        session_id: str,
        completed_file: str,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> None:
        session = self.sessions.get(session_id)
        if session is None:
            logger.debug("Session %s 不存在，自动创建（兼容模式）。", session_id)
            self.create_session(session_id, "unknown", 0, {})
            session = self.sessions[session_id]

        if success:
            session["completed_items"] += 1
            session["processed_files"].append({"file": completed_file, "timestamp": time.time()})
        else:
            session["failed_items"] += 1
            session["failed_files"].append(
                {"file": completed_file, "error": error_message, "timestamp": time.time()}
            )

        stage = self._stage_for_operation(session.get("operation_type"))
        if stage:
            self.update_status(completed_file, stage, "done" if success else "failed")

        session["last_update"] = time.time()

    def update_progress(self, session_id: str, increment: int = 1) -> None:
        session = self.sessions.get(session_id)
        if session is None:
            return
        session["completed_items"] += increment
        session["last_update"] = time.time()

    def get_session_progress(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get(session_id)

    def get_remaining_files(self, session_id: str, all_files: Iterable[str]) -> List[str]:
        session = self.sessions.get(session_id)
        stage = self._stage_for_operation(session.get("operation_type")) if session else None

        remaining: List[str] = []
        for file_path in all_files:
            path_obj = Path(file_path)
            if stage and self.should_skip(path_obj, stage):
                continue
            remaining.append(str(path_obj))
        return remaining

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        session = self.sessions.get(session_id)
        if session is None:
            return {}

        total = session["total_items"]
        completed = session["completed_items"]
        failed = session["failed_items"]
        remaining = max(total - completed - failed, 0)
        return {
            "session_id": session_id,
            "operation_type": session.get("operation_type"),
            "status": session.get("status", "running"),
            "total_items": total,
            "completed_items": completed,
            "failed_items": failed,
            "remaining_items": remaining,
            "completion_percentage": (completed / total * 100) if total else 0,
            "failure_percentage": (failed / total * 100) if total else 0,
        }

    def complete_session(self, session_id: str, status: str = "completed") -> None:
        session = self.sessions.get(session_id)
        if session is None:
            return
        session["status"] = status
        session["end_time"] = time.time()
        session["last_update"] = time.time()

    def reactivate_session(self, session_id: str, new_total_items: Optional[int] = None) -> None:
        session = self.sessions.get(session_id)
        if session is None:
            return
        session["status"] = "running"
        if new_total_items and new_total_items > session["total_items"]:
            session["total_items"] = new_total_items
        session.pop("end_time", None)
        session["last_update"] = time.time()

    def fail_session(self, session_id: str, error_message: str) -> None:
        session = self.sessions.get(session_id)
        if session is None:
            return
        session["status"] = "failed"
        session["error_message"] = error_message
        session["end_time"] = time.time()

    def list_sessions(self) -> List[Dict[str, Any]]:
        return list(self.sessions.values())

    def get_processed_files(self, session_id: str) -> List[str]:
        session = self.sessions.get(session_id)
        if session is None:
            return []
        return [item["file"] for item in session.get("processed_files", [])]

    def get_failed_files(self, session_id: str) -> List[Dict[str, Any]]:
        session = self.sessions.get(session_id)
        if session is None:
            return []
        return session.get("failed_files", [])

    def is_file_processed(self, session_id: str, file_path: str) -> bool:
        processed = self.get_processed_files(session_id)
        return str(file_path) in processed

    def get_progress_summary(self, session_id: str) -> str:
        stats = self.get_session_stats(session_id)
        if not stats:
            return f"Session {session_id} not found"

        summary = [
            f"会话 {session_id}:",
            f"  状态: {stats.get('status', 'unknown')}",
            f"  进度: {stats.get('completed_items', 0)}/{stats.get('total_items', 0)}"
            f" ({stats.get('completion_percentage', 0):.1f}%)",
        ]
        failed = stats.get("failed_items", 0)
        if failed:
            summary.append(f"  失败: {failed}")
        return "\n".join(summary)

    # ------------------------------------------------------------------#
    # 辅助方法
    # ------------------------------------------------------------------#
    def reset_file_stages(self, file_path: Path | str) -> None:
        """手动重置某个文件的所有阶段为 pending。"""
        rel_path = self._relative_path(Path(file_path).resolve())
        record = self.records.get(rel_path)
        if record is None:
            return
        record["stages"] = self._default_stage_states()
        self.records[rel_path] = record
        self._persist()