import os
import tempfile
import uuid
import re
import numpy as np
from typing import List, Any, Tuple
import streamlit as st
from dotenv import load_dotenv

# v1.0 核心导入
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import HumanMessage, AIMessage

# v1.0 组件
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_compressors import FlashrankRerank


# 优化的混合检索器
class HybridParentRetriever(BaseRetriever):
    """v1.0 混合检索器：向量检索(子文档) + BM25(父文档) + 父文档映射"""

    vectorstore: Chroma
    docstore: InMemoryStore
    bm25_docs: List[Document]
    k1: int = 6  # 向量检索子文档数量
    k2: int = 4  # BM25 检索父文档数量

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
        return self.invoke(query, config=run_manager.config if run_manager else None)

    def invoke(self, query: str, config: Any = None, **kwargs: Any) -> List[Document]:
        # 1. 向量检索子文档
        child_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k1})
        child_docs = child_retriever.invoke(query, config=config)

        # 2. BM25 检索父文档
        kw_docs = bm25_search_docs(query, self.bm25_docs, top_k=self.k2)

        # 3. ✅ 修复：正确处理 mget 返回值
        parent_docs_from_vector = []
        for child_doc in child_docs:
            parent_id = child_doc.metadata.get("parent_id")
            if parent_id:
                # mget 返回 List[Optional[Document]]，取第一个有效值
                parent_docs = self.docstore.mget([parent_id])
                if parent_docs and len(parent_docs) > 0 and parent_docs[0] is not None:
                    parent_docs_from_vector.append(parent_docs[0])

        # 4. 去重逻辑保持不变
        all_docs = parent_docs_from_vector + kw_docs
        seen_doc_ids = set()
        unique_parent_docs = []

        for doc in all_docs:
            doc_id = doc.metadata.get("doc_id")
            if doc_id and doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                unique_parent_docs.append(doc)

        return unique_parent_docs[:self.k1 + self.k2]



# ============ 全局配置 ============
CHROMA_PERSIST_DIRECTORY = "./chroma_db_parent"
EMB_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
MAX_HISTORY_LENGTH = 10


# ============ 初始化页面 ============
def init_page() -> None:
    st.set_page_config(page_title="Pro RAG (LangChain v1.0)", page_icon="🚀", layout="wide")

    st.markdown("""
        <style>
        .main-header {font-size: 1.8rem; font-weight: 700; color: #1f77b4;}
        .stChatInput {position: fixed; bottom: 20px;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🚀 Pro RAG v1.0: Hybrid Search + FlashRank</div>',
                unsafe_allow_html=True)

    # 初始化 Session State
    if "docstore" not in st.session_state:
        st.session_state["docstore"] = InMemoryStore()
    if "bm25" not in st.session_state:
        st.session_state["bm25"] = None
    if "bm25_docs" not in st.session_state:
        st.session_state["bm25_docs"] = []
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []


# ============ 核心组件 ============
@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return HuggingFaceEmbeddings(
        model_name=EMB_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
        model_kwargs={"device": "cpu"}
    )


def get_vectorstore() -> Chroma:
    embedding = get_embeddings()

    vectorstore = Chroma(
        collection_name="split_parents",
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_function=embedding
    )

    # 如果 docstore 为空但 chroma 有数据，重置
    if len(list(st.session_state["docstore"].yield_keys())) == 0 and vectorstore._collection.count() > 0:
        st.warning("⚠️ DocStore 与 Chroma 不一致，正在重置...")
        vectorstore.reset_collection()

    return vectorstore


def get_hybrid_retriever() -> HybridParentRetriever:
    """v1.0 混合检索器构建"""
    if not st.session_state["bm25_docs"]:
        return None

    return HybridParentRetriever(
        vectorstore=get_vectorstore(),
        docstore=st.session_state["docstore"],
        bm25_docs=st.session_state["bm25_docs"]
    )


# ============ BM25 实现 ============
def _tokenize_zh(text: str) -> List[str]:
    return re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text)


def rebuild_bm25(docs: List[Document]):
    """重建 BM25 索引"""
    if not docs:
        st.session_state["bm25"] = None
        return

    try:
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [_tokenize_zh(d.page_content) for d in docs]
        st.session_state["bm25"] = BM25Okapi(tokenized_corpus)
        st.session_state["bm25_docs"] = docs
    except ImportError:
        st.error("请安装: pip install rank_bm25")
        st.stop()


def bm25_search_docs(query: str, docs: List[Document], top_k: int = 4) -> List[Document]:
    """BM25 检索"""
    bm25 = st.session_state.get("bm25")
    if not bm25 or not docs:
        return docs[:top_k]  # 降级到简单切片

    tokenized_query = _tokenize_zh(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = [docs[i] for i in top_indices if scores[i] > 0]
    return results[:top_k]


# ============ 文档处理 ============
def ingest_files(uploaded_files):
    """文档索引 - v1.0 修复版"""
    if not uploaded_files:
        return

    raw_docs = []
    with st.status("📥 正在处理文档...", expanded=True) as status:
        for file in uploaded_files:
            status.write(f"解析文件: {file.name}")
            suffix = os.path.splitext(file.name)[1].lower()

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file.getbuffer())
                tmp_path = tmp.name

            try:
                if suffix == ".pdf":
                    loader = PyPDFLoader(tmp_path)
                    raw_docs.extend(loader.load())
                elif suffix in [".txt", ".md"]:
                    loader = TextLoader(tmp_path, encoding="utf-8")
                    raw_docs.extend(loader.load())
                elif suffix in [".docx", ".doc"]:
                    try:
                        import docx2txt
                        text = docx2txt.process(tmp_path)
                        raw_docs.append(Document(
                            page_content=text,
                            metadata={"source": file.name}
                        ))
                    except ImportError:
                        st.warning("docx2txt 未安装，跳过 DOCX 文件")
                        continue
            except Exception as e:
                st.error(f"❌ 解析失败 {file.name}: {e}")
            finally:
                try:
                    os.remove(tmp_path)
                except:
                    pass

        if not raw_docs:
            st.error("❌ 未解析到任何文档内容")
            return

        status.write("🔨 构建知识库索引...")

        # 1. 父文档分割 (大块)
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200
        )
        parent_docs = parent_splitter.split_documents(raw_docs)

        # 2. 添加元数据到父文档
        doc_ids = [str(uuid.uuid4()) for _ in parent_docs]
        parent_doc_map = {}  # 新增：父文档映射表

        for doc, doc_id in zip(parent_docs, doc_ids):
            doc.metadata.update({
                "doc_id": doc_id,
                "source": doc.metadata.get("source", "unknown")
            })
            parent_doc_map[doc_id] = doc  # 存储父文档映射

        # 3. 存储父文档到 docstore
        docstore_pairs = [(doc.metadata["doc_id"], doc) for doc in parent_docs]
        st.session_state["docstore"].mset(docstore_pairs)

        # 4. ✅ 修复：创建子文档 + 明确 parent_id 映射
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        child_docs = []

        for parent_doc in parent_docs:
            parent_id = parent_doc.metadata["doc_id"]
            parent_content = parent_doc.page_content

            # 分割父文档内容为子文档
            child_texts = child_splitter.split_text(parent_content)

            for i, child_text in enumerate(child_texts):
                child_doc = Document(
                    page_content=child_text,
                    metadata={
                        "parent_id": parent_id,  # ✅ 明确关联父文档
                        "doc_id": f"child_{parent_id}_{i}",
                        "source": parent_doc.metadata["source"],
                        "chunk_index": i,
                        "parent_source": parent_doc.metadata["source"]
                    }
                )
                child_docs.append(child_doc)

        # 5. 构建向量索引
        vectorstore = get_vectorstore()
        vectorstore.add_documents(child_docs)

        # 6. 更新 BM25（只用父文档）
        rebuild_bm25(parent_docs)  # 只传入父文档，不累积

        st.success(f"✅ 索引完成！父文档: {len(parent_docs)}, 子文档: {len(child_docs)}")

        # 重置聊天历史
        st.session_state["messages"] = []
        st.session_state["chat_history"] = []


# ============ LLM 配置 ============
def get_llm_model(provider: str, model_name: str, temp: float = 0.1):
    load_dotenv()

    api_key = None
    base_url = None

    if provider == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY")
    elif provider == "DeepSeek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = "https://api.deepseek.com"
    elif provider == "Qwen":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    if not api_key:
        return None

    return ChatOpenAI(
        model=model_name,
        temperature=temp,
        api_key=api_key.strip(),
        base_url=base_url,
        streaming=True
    )


def rewrite_query(original_query: str, history: List, llm: ChatOpenAI) -> str:
    """查询重写"""
    if len(history) < 2:
        return original_query

    system_prompt = (
        "你是一个搜索优化专家。根据对话历史和用户最新问题，"
        "重写为一个独立、完整、包含必要上下文的搜索查询。"
        "只输出查询字符串，不要添加解释。"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()
    try:
        return chain.invoke({
            "history": history[-6:],
            "question": original_query
        })
    except:
        return original_query


def build_rag_chain(llm):
    """RAG 链"""
    qa_system_prompt = (
        "你是一个专业的智能助手。请结合【对话历史】和【参考资料】回答用户的问题。\n"
        "规则：\n"
        "1. 优先依据【参考资料】回答\n"
        "2. 结合【对话历史】保持上下文连贯\n"
        "3. 无法回答时诚实说不知道\n\n"
        "【参考资料】:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}")
    ])

    return prompt | llm | StrOutputParser()


# ============ 主程序 ============
def main():
    init_page()

    with st.sidebar:
        st.header("⚙️ 配置")
        provider = st.selectbox("模型厂商", ["DeepSeek", "OpenAI", "Qwen"])
        default_models = {"DeepSeek": "deepseek-chat", "OpenAI": "gpt-4o-mini", "Qwen": "qwen-turbo"}
        model_name = st.text_input("模型名称", value=default_models[provider])

        st.divider()
        uploaded_files = st.file_uploader(
            "📂 上传知识库",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'md', 'docx']
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 开始索引", type="primary", use_container_width=True):
                ingest_files(uploaded_files)
        with col2:
            if st.button("🗑️ 清空数据", type="secondary", use_container_width=True):
                # 清空所有状态
                st.session_state["docstore"] = InMemoryStore()
                st.session_state["bm25"] = None
                st.session_state["bm25_docs"] = []
                st.session_state["messages"] = []
                st.session_state["chat_history"] = []
                try:
                    get_vectorstore().reset_collection()
                    st.success("✅ 已清空所有数据")
                except:
                    pass
                st.rerun()

        st.info(f"📊 知识库文档: {len(st.session_state.get('bm25_docs', []))}")

    # 聊天历史显示
    for msg_idx, msg in enumerate(st.session_state["messages"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("rewrite"):
                st.caption(f"🔍 优化查询: {msg['rewrite']}")

            # ✅ 修复：在聊天历史显示部分
            if msg.get("sources"):
                with st.expander(f"📚 参考资料 ({len(msg['sources'])} 份)"):
                    for i, doc in enumerate(msg["sources"]):
                        st.markdown(f"**[{i + 1}] {doc.metadata.get('source', '未知')}**")

                        doc_id = doc.metadata.get('doc_id', 'N/A')
                        parent_id = doc.metadata.get('parent_id', 'N/A')
                        st.caption(f"ID: {doc_id} | 父文档: {parent_id}")

                        preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                        # ✅ 修复：使用消息索引+文档索引作为稳定 key
                        st.text_area(
                            f"preview_history_{msg_idx}_{i}",
                            preview,
                            height=100
                        )

    # 用户输入处理
    if user_input := st.chat_input("请输入您的问题..."):
        llm = get_llm_model(provider, model_name)
        if not llm:
            st.error("❌ 请配置正确的 API Key (.env 文件或环境变量)")
            st.stop()

        # 添加用户消息到历史
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # AI 响应
        # AI 响应部分 - 修复版
        with st.chat_message("assistant"):
            status_container = st.status("🤔 正在思考...")
            final_response = ""


            try:
                # 1. 查询优化
                status_container.write("🔄 理解上下文...")
                rewritten_query = rewrite_query(
                    user_input,
                    st.session_state["chat_history"],
                    llm
                )
                if rewritten_query != user_input:
                    status_container.write(f"🔍 优化查询: `{rewritten_query}`")

                # 2. 混合检索 - 先完整执行，再显示调试信息
                status_container.write("📥 执行混合检索...")
                hybrid_retriever = get_hybrid_retriever()
                if not hybrid_retriever:
                    st.error("❌ 请先上传并索引文档！")
                    return

                # 执行完整检索
                raw_docs = hybrid_retriever.invoke(rewritten_query)

                # 3. FlashRank 重排序
                status_container.write("⚖️ FlashRank 智能排序...")
                try:
                    reranker = FlashrankRerank(top_n=4)
                    source_documents = reranker.compress_documents(
                        raw_docs, [Document(page_content=rewritten_query)]
                    )
                except:
                    source_documents = raw_docs[:4]

                # ✅ 现在 source_documents 已正确赋值，再显示调试信息
                status_container.write(f"✅ 检索到 {len(source_documents)} 份高质量资料")
                status_container.write("🔍 检索详情:")

                # 获取调试用的子文档和BM25文档（用于显示，不影响source_documents）
                child_retriever = hybrid_retriever.vectorstore.as_retriever(
                    search_kwargs={"k": hybrid_retriever.k1}
                )
                child_docs = child_retriever.invoke(rewritten_query)
                kw_docs = bm25_search_docs(
                    rewritten_query,
                    hybrid_retriever.bm25_docs,
                    top_k=hybrid_retriever.k2
                )

                status_container.write(f"  - 向量子文档: {len(child_docs)}")
                status_container.write(f"  - BM25父文档: {len(kw_docs)}")
                status_container.write(f"  - FlashRank后: {len(source_documents)}")
                status_container.write(
                    f"  - 示例文档: {[d.metadata.get('source', 'N/A')[:30] for d in source_documents[:2]]}")

                # 4. 生成回答
                status_container.write("💭 生成智能回答...")
                context_text = "\n\n".join([d.page_content for d in source_documents])
                chain = build_rag_chain(llm)

                input_dict = {
                    "context": context_text,
                    "question": user_input,
                    "chat_history": st.session_state["chat_history"][-6:]
                }

                placeholder = st.empty()
                for chunk in chain.stream(input_dict):
                    final_response += chunk
                    placeholder.markdown(final_response + "▌")
                placeholder.markdown(final_response)

                status_container.update(label="✅ 完成！", state="complete")

                # 🔥 新增：立即显示当前消息的参考资料
                if source_documents:
                    # ❌ 删除 expanded=True 参数
                    with st.expander(f"📚 参考资料 ({len(source_documents)} 份)"):  # ✅ 已修复
                        for i, doc in enumerate(source_documents):
                            st.markdown(f"**[{i + 1}] {doc.metadata.get('source', '未知')}**")

                            doc_id = doc.metadata.get('doc_id', 'N/A')
                            parent_id = doc.metadata.get('parent_id', 'N/A')
                            st.caption(f"ID: {doc_id} | 父文档: {parent_id}")

                            preview = doc.page_content[:500] + "..." if len(
                                doc.page_content) > 500 else doc.page_content
                            # ✅ 同时修复 key 冲突
                            current_msg_idx = len(st.session_state["messages"])
                            st.text_area(
                                f"preview_current_{current_msg_idx}_{i}",  # ✅ 稳定 key
                                preview,
                                height=100
                            )

            except Exception as e:
                # ... 错误处理保持不变 ...
                return

            # 然后才更新会话状态（现有代码）
            st.session_state["messages"].append({
                "role": "assistant",
                "content": final_response,
                "rewrite": rewritten_query if rewritten_query != user_input else None,
                "sources": source_documents  # 这个用于历史显示
            })

        st.session_state["chat_history"].extend([
            HumanMessage(content=user_input),
            AIMessage(content=final_response)
        ])

        # 控制历史长度
        if len(st.session_state["chat_history"]) > MAX_HISTORY_LENGTH * 2:
            st.session_state["chat_history"] = st.session_state["chat_history"][-MAX_HISTORY_LENGTH * 2:]




if __name__ == "__main__":
    main()
