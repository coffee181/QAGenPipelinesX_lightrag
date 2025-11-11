#!/usr/bin/env python3
"""
QA生成管道后端集成示例

此文件展示如何在后端系统中集成QA生成管道可执行文件，
通过子进程调用的方式实现QA对生成功能。
"""

import subprocess
import json
import os
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

@dataclass
class QAGenerationResult:
    """QA生成结果"""
    success: bool
    qa_pairs_count: int = 0
    output_file: Optional[str] = None
    session_id: Optional[str] = None
    error: Optional[str] = None

class QAGenerationClient:
    """QA生成管道客户端
    
    通过子进程调用可执行文件实现QA生成功能
    """
    
    def __init__(self, executable_path: str = "./deployment/qa_gen_pipeline"):
        """初始化客户端
        
        Args:
            executable_path: 可执行文件路径
                Windows: "./deployment/qa_gen_pipeline.exe"
                Linux/macOS: "./deployment/qa_gen_pipeline"
        """
        self.executable_path = executable_path
        
        # 检查可执行文件是否存在
        if not os.path.exists(executable_path):
            raise FileNotFoundError(f"可执行文件不存在: {executable_path}")
    
    def generate_answers_from_existing_kb(
        self,
        questions_file: Union[str, Path],
        working_dir: Union[str, Path], 
        output_file: Union[str, Path],
        session_id: Optional[str] = None,
        restart: bool = False
    ) -> QAGenerationResult:
        """从现有知识库生成答案
        
        Args:
            questions_file: 问题文件路径
            working_dir: 知识库工作目录
            output_file: 输出文件路径
            session_id: 会话ID
            restart: 是否强制重新开始
            
        Returns:
            QAGenerationResult: 生成结果
        """
        cmd = [
            self.executable_path,
            "generate-answers",
            str(questions_file),
            str(working_dir),
            str(output_file)
        ]
        
        if session_id:
            cmd.extend(["--session-id", session_id])
            
        if restart:
            cmd.append("--restart")
        
        return self._execute_command(cmd, session_id)
    
    def generate_answers_with_documents(
        self,
        questions_file: Union[str, Path],
        working_dir: Union[str, Path],
        output_file: Union[str, Path], 
        documents_path: Union[str, Path],
        session_id: Optional[str] = None,
        restart: bool = False
    ) -> QAGenerationResult:
        """带文档插入的答案生成
        
        Args:
            questions_file: 问题文件路径
            working_dir: 知识库工作目录
            output_file: 输出文件路径
            documents_path: 要插入的文档路径
            session_id: 会话ID
            restart: 是否强制重新开始
            
        Returns:
            QAGenerationResult: 生成结果
        """
        cmd = [
            self.executable_path,
            "generate-answers",
            str(questions_file),
            str(working_dir),
            str(output_file),
            "-i", str(documents_path)
        ]
        
        if session_id:
            cmd.extend(["--session-id", session_id])
            
        if restart:
            cmd.append("--restart")
        
        return self._execute_command(cmd, session_id)
    
    def insert_documents(
        self,
        working_dir: Union[str, Path],
        documents_path: Union[str, Path],
        session_id: Optional[str] = None
    ) -> QAGenerationResult:
        """插入文档到知识库
        
        Args:
            working_dir: 知识库工作目录
            documents_path: 要插入的文档路径
            session_id: 会话ID
            
        Returns:
            QAGenerationResult: 插入结果
        """
        cmd = [
            self.executable_path,
            "insert-documents",
            str(working_dir),
            str(documents_path)
        ]
        
        if session_id:
            cmd.extend(["--session-id", session_id])
        
        return self._execute_command(cmd, session_id)
    
    def get_progress(self, session_id: str) -> Optional[Dict]:
        """获取会话进度
        
        Args:
            session_id: 会话ID
            
        Returns:
            进度信息字典，如果会话不存在返回None
        """
        cmd = [
            self.executable_path,
            "show-progress",
            "--session-id", session_id
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(self.executable_path),
                timeout=30
            )
            
            if result.returncode != 0:
                return None
                
            return self._parse_progress(result.stdout)
            
        except (subprocess.TimeoutExpired, Exception):
            return None
    
    def _execute_command(self, cmd: List[str], session_id: Optional[str] = None) -> QAGenerationResult:
        """执行命令
        
        Args:
            cmd: 命令参数列表
            session_id: 会话ID
            
        Returns:
            QAGenerationResult: 执行结果
        """
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(self.executable_path),
                timeout=3600  # 1小时超时
            )
            
            if result.returncode != 0:
                return QAGenerationResult(
                    success=False,
                    error=result.stderr or "未知错误"
                )
            
            # 解析输出
            qa_pairs_count = self._extract_qa_count(result.stdout)
            
            return QAGenerationResult(
                success=True,
                qa_pairs_count=qa_pairs_count,
                session_id=session_id
            )
            
        except subprocess.TimeoutExpired:
            return QAGenerationResult(
                success=False,
                error="操作超时"
            )
        except Exception as e:
            return QAGenerationResult(
                success=False,
                error=str(e)
            )
    
    def _extract_qa_count(self, output: str) -> int:
        """从输出中提取QA对数量"""
        import re
        for line in output.split('\n'):
            if 'Generated' in line and 'QA pairs' in line:
                match = re.search(r'Generated (\d+) QA pairs', line)
                if match:
                    return int(match.group(1))
        return 0
    
    def _parse_progress(self, output: str) -> Dict:
        """解析进度输出"""
        import re
        progress_data = {"completion_percentage": 0.0}
        
        if "完成率" in output:
            match = re.search(r'完成率: ([\d.]+)%', output)
            if match:
                progress_data["completion_percentage"] = float(match.group(1))
        
        return progress_data

class QAGenerationService:
    """QA生成服务
    
    提供高级的QA生成服务接口，包括文件管理和结果处理
    """
    
    def __init__(self, client: QAGenerationClient, temp_dir: str = "./temp"):
        """初始化服务
        
        Args:
            client: QA生成客户端
            temp_dir: 临时文件目录
        """
        self.client = client
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
    
    def generate_qa_from_questions(
        self,
        questions: List[str],
        knowledge_base_dir: Union[str, Path],
        request_id: str,
        include_documents: Optional[Union[str, Path]] = None
    ) -> Dict:
        """从问题列表生成QA对
        
        Args:
            questions: 问题列表
            knowledge_base_dir: 知识库目录
            request_id: 请求ID
            include_documents: 要包含的文档路径（可选）
            
        Returns:
            生成结果字典
        """
        # 创建临时文件
        questions_file = self.temp_dir / f"questions_{request_id}.jsonl"
        output_file = self.temp_dir / f"qa_output_{request_id}.jsonl"
        
        # 保存问题到文件
        with open(questions_file, 'w', encoding='utf-8') as f:
            for question in questions:
                f.write(json.dumps({"question": question}, ensure_ascii=False) + '\n')
        
        try:
            # 生成答案
            if include_documents:
                result = self.client.generate_answers_with_documents(
                    questions_file=questions_file,
                    working_dir=knowledge_base_dir,
                    output_file=output_file,
                    documents_path=include_documents,
                    session_id=f"service_{request_id}"
                )
            else:
                result = self.client.generate_answers_from_existing_kb(
                    questions_file=questions_file,
                    working_dir=knowledge_base_dir,
                    output_file=output_file,
                    session_id=f"service_{request_id}"
                )
            
            if not result.success:
                return {
                    "success": False,
                    "error": result.error,
                    "request_id": request_id
                }
            
            # 读取生成的QA对
            qa_pairs = []
            if output_file.exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            qa_data = json.loads(line)
                            qa_pairs.append(qa_data)
            
            return {
                "success": True,
                "request_id": request_id,
                "qa_pairs": qa_pairs,
                "qa_pairs_count": result.qa_pairs_count,
                "session_id": result.session_id
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "request_id": request_id
            }
        finally:
            # 清理临时文件
            try:
                if questions_file.exists():
                    questions_file.unlink()
            except:
                pass
    
    def monitor_progress(self, session_id: str) -> Dict:
        """监控生成进度
        
        Args:
            session_id: 会话ID
            
        Returns:
            进度信息
        """
        progress = self.client.get_progress(session_id)
        if progress is None:
            return {
                "success": False,
                "error": "会话不存在或已完成"
            }
        
        return {
            "success": True,
            "session_id": session_id,
            **progress
        }

# 使用示例
def example_backend_integration():
    """后端集成使用示例"""
    
    # 1. 初始化客户端（根据操作系统选择可执行文件）
    import platform
    if platform.system() == "Windows":
        executable_path = "./deployment/qa_gen_pipeline.exe"
    else:
        executable_path = "./deployment/qa_gen_pipeline"
    
    try:
        client = QAGenerationClient(executable_path)
        service = QAGenerationService(client)
        
        # 2. 示例用户请求
        user_request = {
            "request_id": "backend_001",
            "questions": [
                "如何配置GSK数控系统的参数？",
                "数控系统的安全注意事项有哪些？",
                "如何进行数控系统的日常维护？"
            ]
        }
        
        print(f"处理请求: {user_request['request_id']}")
        print(f"问题数量: {len(user_request['questions'])}")
        
        # 3. 生成QA对
        result = service.generate_qa_from_questions(
            questions=user_request['questions'],
            knowledge_base_dir="./working",
            request_id=user_request['request_id']
        )
        
        # 4. 处理结果
        if result['success']:
            print(f"✓ 生成成功")
            print(f"  - QA对数量: {result['qa_pairs_count']}")
            print(f"  - 会话ID: {result['session_id']}")
            
            # 显示生成的QA对
            for i, qa_pair in enumerate(result['qa_pairs'][:2], 1):  # 只显示前2个
                messages = qa_pair.get('messages', [])
                if len(messages) >= 2:
                    print(f"\n  QA对 {i}:")
                    print(f"    问题: {messages[0]['content'][:50]}...")
                    print(f"    答案: {messages[1]['content'][:100]}...")
        else:
            print(f"❌ 生成失败: {result['error']}")
        
        return result
        
    except FileNotFoundError as e:
        print(f"❌ 可执行文件未找到: {e}")
        print("请先运行 'python build_executable.py' 创建可执行文件")
        return None
    except Exception as e:
        print(f"❌ 意外错误: {e}")
        return None

# Flask API集成示例
def create_flask_api():
    """创建Flask API示例"""
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("Flask未安装，跳过API示例")
        return None
    
    app = Flask(__name__)
    
    # 初始化服务
    import platform
    executable_path = "./deployment/qa_gen_pipeline.exe" if platform.system() == "Windows" else "./deployment/qa_gen_pipeline"
    
    try:
        client = QAGenerationClient(executable_path)
        service = QAGenerationService(client)
    except FileNotFoundError:
        print("可执行文件未找到，API服务无法启动")
        return None
    
    @app.route('/api/generate-qa', methods=['POST'])
    def api_generate_qa():
        """生成QA对API端点"""
        data = request.json
        
        # 验证请求
        if not data or 'questions' not in data or 'request_id' not in data:
            return jsonify({
                "success": False,
                "error": "缺少必需字段: questions, request_id"
            }), 400
        
        # 生成QA对
        result = service.generate_qa_from_questions(
            questions=data['questions'],
            knowledge_base_dir=data.get('knowledge_base_dir', './working'),
            request_id=data['request_id'],
            include_documents=data.get('include_documents')
        )
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 500
    
    @app.route('/api/progress/<session_id>', methods=['GET'])
    def api_get_progress(session_id):
        """获取进度API端点"""
        result = service.monitor_progress(session_id)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 404
    
    @app.route('/api/health', methods=['GET'])
    def api_health():
        """健康检查API端点"""
        return jsonify({
            "status": "healthy",
            "service": "QA Generation API",
            "executable_path": executable_path
        })
    
    return app

if __name__ == "__main__":
    print("🚀 QA生成管道后端集成示例")
    print("=" * 50)
    
    # 运行基本示例
    result = example_backend_integration()
    
    if result:
        print("\n" + "=" * 50)
        print("💡 提示:")
        print("- 可执行文件路径需要根据实际部署情况调整")
        print("- 知识库目录需要预先存在并包含文档")
        print("- API密钥需要通过环境变量或.env文件配置")
        print("- 更多使用方法请参考 Docs/API_USAGE_GUIDE.md") 