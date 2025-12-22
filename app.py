import os
import shutil
import tempfile
from typing import List

import streamlit as st
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # 新包名，LangChain 1.0+ 推荐

from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek

from langchain_community.document_loaders import PyPDFLoader, TextLoader
import docx2txt


# ============ 全局常量配置 ============
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
EMB_MODEL_NAME = "BAAI/bge-large-zh-v1.5"  # 升级为更强的中文模型


def init_page() -> None:
    st.set_page_config(page_title="中文 RAG 知识库助手", page_icon="📚", layout="wide")

    st.markdown(
        """
        <style>
        .main-header {font-size: 2.0rem; font-weight: 700; margin-bottom: 0.3rem;}
        .sub-header {font-size: 0.95rem; color: #666666; margin-bottom: 1.2rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-header">📚 中文 RAG 知识库助手（Qwen / DeepSeek + Chroma）</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">上传 PDF / DOCX / TXT 文档，构建本地向量知识库，使用云端大模型进行高质量问答。</div>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="正在加载中文嵌入模型（仅首次较慢）...")
def get_embeddings() -> HuggingFaceEmbeddings:
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return HuggingFaceEmbeddings(
        model_name=EMB_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore() -> Chroma:
    """统一获取（或创建）向量库实例，支持增量添加。"""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_function=embeddings,
    )


def ingest_docs(uploaded_files: List) -> int:
    """加载、切分并增量写入向量库，返回本次新增的 chunk 数量。"""
    if not uploaded_files:
        return 0

    raw_docs: List[Document] = []
    temp_dirs = []  # 收集临时目录，便于清理

    for f in uploaded_files:
        suffix = os.path.splitext(f.name)[1].lower()
        temp_dir = tempfile.mkdtemp(prefix="rag_upload_")
        temp_dirs.append(temp_dir)
        file_path = os.path.join(temp_dir, f.name)

        with open(file_path, "wb") as out_f:
            out_f.write(f.getbuffer())

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            elif suffix in [".txt", ".md"]:
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
            elif suffix in [".docx", ".doc"]:
                text = docx2txt.process(file_path)
                if text.strip():
                    docs = [Document(page_content=text, metadata={"source": f.name})]
                else:
                    docs = []
            else:
                st.warning(f"不支持的文件格式：{suffix}，已跳过 {f.name}")
                continue

            for d in docs:
                d.metadata.setdefault("source", f.name)
            raw_docs.extend(docs)
        except Exception as e:
            st.error(f"加载 {f.name} 时出错：{e}")

    if not raw_docs:
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    )
    split_docs = splitter.split_documents(raw_docs)

    # 增量添加：使用 from_documents 会自动处理现有集合
    vectorstore = get_vectorstore()
    added_ids = vectorstore.add_documents(split_docs)  # 返回新增的 ID

    # 清理临时文件
    for td in temp_dirs:
        shutil.rmtree(td, ignore_errors=True)

    return len(added_ids)


def get_retriever():
    """获取检索器（如果库为空返回 None）。"""
    vectorstore = get_vectorstore()
    if vectorstore._collection.count() == 0:
        return None
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def get_llm(provider: str, model_name: str):
    load_dotenv(override=False)

    if provider == "Qwen (通义千问)":
        api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            st.error("未检测到 DASHSCOPE_API_KEY，请在 .env 中配置。")
            return None
        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            streaming=True,
        )

    elif provider == "DeepSeek":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            st.error("未检测到 DEEPSEEK_API_KEY，请在 .env 中配置。")
            return None
        return ChatDeepSeek(
            model=model_name,
            api_key=api_key,
            streaming=True,
        )

    st.error("未知的 LLM 提供方。")
    return None


def build_rag_chain(retriever, llm):
    system_prompt = (
        "你是一名专业的中文 AI 助手，基于提供的知识库内容回答用户问题。\n"
        "要求：\n"
        "1. 回答必须使用地道、流畅的中文。\n"
        "2. 严格依据 context 中的信息推理，禁止胡编乱造。\n"
        "3. 如果 context 不足以回答，请明确说“知识库中暂无足够信息”，并给出合理推测。\n"
        "4. 回答结构清晰，可使用分点、列表等格式。"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "相关知识库内容：\n{context}\n\n问题：{question}"),
        ]
    )

    def format_docs(docs: List[Document]) -> str:
        return "\n\n---\n\n".join(
            f"[来源：{d.metadata.get('source', '未知')}] {d.page_content}" for d in docs
        )

    rag_chain = (
        RunnableParallel(
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def display_sources(query: str, retriever) -> None:
    if retriever is None:
        return

    docs = retriever.invoke(query)
    if not docs:
        st.info("本次提问未检索到相关文档片段。")
        return

    st.markdown("---")
    st.subheader("📎 检索到的参考文档片段")

    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", f"片段 {i}")
        with st.expander(f"📄 {source}（片段 {i}）", expanded=False):
            st.markdown(doc.page_content)


def sidebar_controls() -> tuple[str, str, List]:
    with st.sidebar:
        st.header("⚙️ 配置")

        provider = st.selectbox("选择大模型提供方", ["Qwen (通义千问)", "DeepSeek"], index=0)

        default_model = "qwen-plus" if provider == "Qwen (通义千问)" else "deepseek-chat"
        model_name = st.text_input("模型名称", value=default_model)

        st.markdown("---")
        st.subheader("📁 上传文档（增量构建知识库）")
        uploaded_files = st.file_uploader(
            "支持 PDF / DOCX / TXT / MD，多文件上传",
            accept_multiple_files=True,
            type=["pdf", "txt", "docx", "doc", "md"],
        )

        if st.button("🚀 开始索引文档", use_container_width=True):
            if not uploaded_files:
                st.warning("请先上传文档。")
            else:
                with st.spinner("正在处理并向量化文档..."):
                    added = ingest_docs(uploaded_files)
                if added > 0:
                    st.success(f"本次成功新增 {added} 个文档片段。")
                else:
                    st.warning("未新增任何片段，请检查文件内容。")

        st.markdown("---")
        st.caption(
            "✅ 向量库：Chroma 本地持久化\n"
            "✅ 嵌入模型：BAAI/bge-large-zh-v1.5（本地推理，无需 API）"
        )

    return provider, model_name, uploaded_files


def render_chat_area(provider: str, model_name: str) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("输入问题，按回车发送...")
    if not user_query:
        return

    retriever = get_retriever()
    if retriever is None:
        st.warning("知识库为空，请先上传并索引文档。")
        return

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    llm = get_llm(provider, model_name)
    if llm is None:
        return

    rag_chain = build_rag_chain(retriever, llm)

    with st.chat_message("assistant"):
        response = st.write_stream(rag_chain.stream(user_query))
        st.session_state.messages.append({"role": "assistant", "content": response})

    display_sources(user_query, retriever)


def main() -> None:
    load_dotenv(override=False)
    init_page()
    provider, model_name, _ = sidebar_controls()
    render_chat_area(provider, model_name)


if __name__ == "__main__":
    main()
