# 业务逻辑更新：默认恢复行为

## 更改概述

根据用户需求，我们修改了QA生成管道的默认行为：

### 之前的行为
- **默认**: 从头开始生成答案
- **参数**: 使用 `--resume/-r` 参数启用恢复功能

### 更新后的行为
- **默认**: 自动从中断点恢复（如果存在进度文件）
- **参数**: 使用 `--restart` 参数强制从头开始

## 具体修改

### 1. 命令行参数变更

#### 旧版本
```bash
# 默认从头开始
python main.py generate-answers questions.jsonl kb_dir output.jsonl

# 恢复模式
python main.py generate-answers questions.jsonl kb_dir output.jsonl --resume
```

#### 新版本
```bash
# 默认恢复模式
python main.py generate-answers questions.jsonl kb_dir output.jsonl

# 强制重新开始
python main.py generate-answers questions.jsonl kb_dir output.jsonl --restart
```

### 2. API 参数变更

#### 旧版本
```python
# 默认参数
answer_service.generate_answers_for_questions(
    questions_file, output_file, session_id, resume=False
)
```

#### 新版本
```python
# 默认参数
answer_service.generate_answers_for_questions(
    questions_file, output_file, session_id, resume=True
)
```

## 用户体验改进

### 优势
1. **更好的容错性**: 程序意外中断后，重新运行命令即可自动恢复
2. **简化操作**: 不需要记住恢复参数，默认行为更符合预期
3. **减少数据丢失**: 避免因忘记加恢复参数而重复已完成的工作

### 行为逻辑
1. **检测进度文件**: 程序启动时自动检查输出文件是否存在
2. **智能恢复**: 如果发现现有进度，自动从中断点继续
3. **强制重新开始**: 只有明确指定 `--restart` 才会忽略现有进度

## 兼容性说明

### 向后兼容
- 现有的脚本和代码仍然可以工作
- 显式指定 `resume=False` 的代码行为不变
- 配置文件格式无变化

### 注意事项
- 如果需要强制从头开始，必须使用 `--restart` 参数
- 清理输出文件仍然是强制重新开始的另一种方式

## 文档更新

以下文档已同步更新：
- `INCREMENTAL_SAVE_README.md` - 使用说明
- `IMPLEMENTATION_SUMMARY.md` - 实现总结
- `example_incremental_save.py` - 示例脚本注释

## 测试建议

建议测试以下场景：
1. 正常生成任务（应该自动检测和恢复）
2. 使用 `--restart` 强制重新开始
3. 不存在进度文件时的行为（应该正常开始新任务）
4. 损坏的进度文件处理（应该自动回退到新任务）

这个更改使得系统更加用户友好，减少了因操作失误导致的时间浪费。 