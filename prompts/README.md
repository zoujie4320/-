# Prompts 模板库

此文件夹用于存放 RAG 系统的 Prompt 模板（Markdown 格式）。

## 文件说明

| 文件 | 用途 |
|------|------|
| `术语提取.md` | 从文档中识别专业术语、缩写及定义 |
| `实体关系抽取.md` | 构建知识图谱的实体关系三元组 |
| `文档摘要生成.md` | 生成结构化文档摘要 |

## 使用方式

这些 Prompt 模板可用于：
1. **预处理阶段**：在文档解析后调用 LLM 提取术语/实体，增强 chunk 元数据
2. **检索阶段**：对搜索结果进行实体增强，提升相关性
3. **后处理阶段**：对检索结果进行摘要/关系抽取

## 添加新模板

直接在 `prompts/` 目录下创建新的 `.md` 文件即可。命名规范：
- 使用中文描述性文件名
- 每个文件包含：角色定义、输入说明、任务描述、输出格式、示例、注意事项

## 在代码中使用

```python
from pathlib import Path

# 加载 prompt 模板
prompt_dir = Path(__file__).parent / "prompts"
term_prompt = (prompt_dir / "术语提取.md").read_text(encoding="utf-8")

# 拼接到 LLM 请求中
full_prompt = term_prompt + "\n\n## 待处理文档\n" + document_text
```
