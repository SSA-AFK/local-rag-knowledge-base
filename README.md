# 中文 RAG 知识库助手（Local Chinese RAG Assistant）

[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green)](https://python.langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个完全本地运行的中文 RAG（检索增强生成）应用，支持上传 PDF、DOCX、TXT、MD 文档，构建私有知识库，使用通义千问（Qwen）或 DeepSeek 进行高质量问答。

**核心亮点（适合写在简历里）**：
- 完整 LangChain 现代 LCEL 链式构建
- 本地 Chroma 向量数据库 + BAAI/bge-large-zh-v1.5 中文嵌入（无需 API）
- 支持文档增量索引、多轮对话、流式输出
- 显示检索来源片段，提升答案可信度
- Streamlit 友好交互界面


## 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/你的用户名/chinese-rag-assistant.git
cd chinese-rag-assistant

# 中文 RAG 知识库助手

一个基于 **Streamlit** 的本地中文 RAG（Retrieval-Augmented Generation）应用，支持上传 PDF、DOCX、TXT、MD 文档，构建本地向量知识库，使用云端大模型（通义千问 Qwen 或 DeepSeek）进行高质量问答。

**核心特点**：
- **本地向量库**：使用 Chroma（持久化到本地 `./chroma_db` 目录），无需外部数据库。
- **中文嵌入模型**：本地运行 `BAAI/bge-large-zh-v1.5`（HuggingFace 开源，无需 API Key），在 C-MTEB 中文基准上表现优秀。
- **支持增量索引**：多次上传文档时自动增量添加，避免重复处理。
- **流式回答**：实时显示大模型生成过程。
- **来源展示**：回答下方显示检索到的相关文档片段，便于验证答案可靠性。
- **完全本地隐私**：文档和向量库均存储在本地，不上传任何数据。

## 截图（示例）
<img width="2560" height="1600" alt="6f799acfb243d8761fbea4d314ae67a9" src="https://github.com/user-attachments/assets/3dfc8962-80a5-446f-aed7-086dbad6bda2" />

（运行后界面效果：左侧上传文档 + 配置模型，右侧聊天问答 + 来源展示）

## 环境要求

- Python 3.9+
- GPU（可选）：如果有 NVIDIA GPU，可加速嵌入模型加载（需安装 torch with cuda）。
- 内存建议：至少 8GB（bge-large-zh-v1.5 模型约 1.3GB）。

## 安装依赖

```bash
pip install streamlit langchain langchain-chroma langchain-huggingface langchain-openai langchain-deepseek langchain-community python-dotenv docx2txt
```

> 注意：`langchain-deepseek` 是社区包，如果未安装可通过 `pip install langchain-deepseek` 获取。

首次运行会自动从 HuggingFace 下载嵌入模型（约 1.3GB），请确保网络畅通。

## 配置 API Key

本应用支持两种云端大模型：

1. **通义千问 (Qwen)**：通过 DashScope 兼容 OpenAI 接口。
2. **DeepSeek**：官方 API。

创建 `.env` 文件（放在项目根目录），内容示例：

```env
DASHSCOPE_API_KEY=sk-your-dashscope-key-here   # 通义千问 API Key，从 https://dashscope.aliyun.com 获取
DEEPSEEK_API_KEY=sk-your-deepseek-key-here    # DeepSeek API Key，从 https://platform.deepseek.com 获取
```

> 至少配置其中一个即可使用对应模型。

## 运行应用

```bash
streamlit run app.py
```

（将优化后的代码保存为 `app.py`）

打开浏览器访问 `http://localhost:8501`，即可使用！

## 使用流程

1. **左侧边栏配置**：
   - 选择大模型提供方（Qwen 或 DeepSeek）。
   - 可修改模型名称（默认 qwen-plus / deepseek-chat，支持 qwen-max、deepseek-reasoner 等）。

2. **上传文档**：
   - 支持多文件上传（PDF、DOCX、DOC、TXT、MD）。
   - 点击 “开始索引文档”，应用会自动读取、切分、向量化并增量存入本地 Chroma 库。
   - 索引过程显示进度，完成后显示新增片段数量。

3. **聊天问答**：
   - 在底部输入框提问，问题会基于知识库进行检索 + 生成。
   - 回答实时流式显示。
   - 回答下方自动展开检索到的参考文档片段（来源 + 内容）。

## 技术栈详情

- **前端**：Streamlit
- **RAG 框架**：LangChain（LCEL 风格链）
- **向量数据库**：Chroma（本地持久化）
- **嵌入模型**：`BAAI/bge-large-zh-v1.5`（本地推理，中文检索效果优秀）
- **文档加载**：
  - PDF：PyPDFLoader
  - TXT/MD：TextLoader
  - DOCX：docx2txt
- **文本切分**：RecursiveCharacterTextSplitter（chunk_size=1000, overlap=200）
- **大模型**：
  - Qwen：兼容 OpenAI 接口（base_url 为 DashScope）
  - DeepSeek：专用 ChatDeepSeek

## 注意事项

- **删除知识库**：如需清空所有文档，直接删除 `./chroma_db` 文件夹后重启应用。
- **模型切换**：切换大模型后无需重新索引文档。
- **性能**：首次加载嵌入模型较慢，后续通过 `@st.cache_resource` 缓存。
- **隐私**：所有文档处理均在本地完成，仅提问时会发送上下文到云端大模型 API。

## 未来改进建议

- 添加删除单个文档功能。
- 支持更多文件格式（如 PPTX、图片 OCR）。
- 集成 reranker 提升检索精度（例如 bge-reranker-large）。
- 添加聊天历史导出。

享受你的私人中文知识库助手！如果有问题，欢迎反馈。
