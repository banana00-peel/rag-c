from dataclasses import dataclass

from app.config.settings import Settings
from app.conversation.contextualizer import DeepSeekQuestionContextualizer
from app.generation.chat_graph import ChatGraphService
from app.indexing.bm25_store import BM25Store
from app.indexing.chroma_store import ChromaStore
from app.indexing.embeddings import OpenAICompatibleEmbeddingProvider
from app.indexing.service import IndexingService
from app.ingestion.chunker import DocumentChunker
from app.ingestion.service import PdfIngestionService
from app.persistence.repository import DocumentRepository
from app.persistence.conversation_repository import ConversationRepository
from app.retrieval.hybrid import HybridRetriever
from app.services.conversation_service import ConversationService
from app.services.document_service import DocumentService


@dataclass
class ServiceContainer:
    settings: Settings
    documents: DocumentRepository
    document_service: DocumentService
    indexing: IndexingService
    retriever: HybridRetriever
    chat: ChatGraphService
    conversation: ConversationService


def build_container(settings: Settings, session_factory) -> ServiceContainer:
    documents = DocumentRepository(session_factory)                     # 创建文档仓库，负责和数据库打交道。
    embedding_provider = OpenAICompatibleEmbeddingProvider(settings)    # 创建 Embedding 服务，用来把文本转成向量。
    chroma = ChromaStore(settings.chroma_dir, embedding_provider)       # 创建 Chroma 向量数据库。
    # BM25 在应用启动恢复阶段统一从 SQLite 分块重建。
    bm25 = BM25Store(settings.bm25_snapshot_path)
    ingestion = PdfIngestionService(settings)   # 创建 PDF 解析服务，读取pdf文件内容。
    chunker = DocumentChunker(settings)     # 文本切块服务
    indexing = IndexingService(
        settings=settings,
        documents=documents,
        ingestion=ingestion,
        chunker=chunker,
        chroma=chroma,
        bm25=bm25,
    ) # 创建索引服务，负责把 PDF 文档处理成可检索的知识库。
    retriever = HybridRetriever(settings, chroma, bm25) # 创建混合检索器。
    chat = ChatGraphService(settings, retriever)
    conversations = ConversationRepository(session_factory)
    conversations.delete_expired()
    contextualizer = DeepSeekQuestionContextualizer(settings)
    return ServiceContainer(
        settings=settings,
        documents=documents,
        document_service=DocumentService(settings, documents),
        indexing=indexing,
        retriever=retriever,
        chat=chat, # 创建单轮问答服务，负责根据检索结果调用 DeepSeek 生成回答。
        conversation=ConversationService(
            settings,
            conversations,
            contextualizer,
            chat,
        ),
    )

