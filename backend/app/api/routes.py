import logging
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status

from app.api.background_task_executor import FastAPIBackgroundTaskExecutor
from app.api.dependencies import get_container
from app.domain.models import CitationSource, RefusalCode
from app.domain.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentMetadataUpdate,
    DocumentResponse,
    HealthResponse,
    SourceResponse,
    UploadResponse,
)
from app.core.container import ServiceContainer
from app.indexing.service import (
    DocumentDeletionError,
    DocumentNotFoundError,
    DocumentStateConflictError,
    IndexTaskSubmissionError,
)
from app.services.document_service import DocumentUploadError

logger = logging.getLogger(__name__)
router = APIRouter()

# 相应数据封装，不把数据库对象直接返回给前端，只返回需要的信息
def to_document_response(document) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        brand=document.brand,
        model=document.model,
        status=document.status,
        page_count=document.page_count,
        chunk_count=document.chunk_count,
        error_message=document.error_message,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


def to_source_response(source: CitationSource) -> SourceResponse:
    context = source.context
    return SourceResponse(
        source_id=source.source_id,
        document_id=context.document_id,
        document_name=context.document_name,
        page_start=context.page_start,
        page_end=context.page_end,
        model=context.model,
        snippet=context.text,
        chunk_ids=list(context.chunk_ids),
        dense_score=context.dense_score,
        dense_distance=context.dense_distance,
        bm25_score=context.bm25_score,
        rrf_score=context.rrf_score,
    )


def build_cited_source_responses(result: dict) -> list[SourceResponse]:
    source_by_id = {
        source.source_id: source for source in result.get("citation_sources", [])
    }
    responses: list[SourceResponse] = []
    for source_id in result.get("cited_source_ids", []):
        source = source_by_id.get(source_id)
        if source is None:
            raise ValueError(f"回答引用的来源 {source_id} 缺少可信元数据。")
        responses.append(to_source_response(source))
    if not result.get("refused", False) and not responses:
        raise ValueError("非拒答响应没有已验证的可信来源。")
    return responses


@router.get("/health", response_model=HealthResponse)
def health(container: ServiceContainer = Depends(get_container)) -> HealthResponse:
    settings = container.settings
    return HealthResponse(
        status="ok",
        deepseek_configured=bool(settings.deepseek_api_key),
        embedding_configured=bool(settings.embedding_api_key),
        indexed_documents=container.documents.count_completed(),
    )


@router.get("/documents", response_model=list[DocumentResponse])
def list_documents(container: ServiceContainer = Depends(get_container)):
    return [to_document_response(document) for document in container.documents.list_all()]

# 文档上传后端接口
@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    container: ServiceContainer = Depends(get_container),
):
    try:
        document, duplicate = await container.document_service.upload(file)
    except DocumentUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return UploadResponse(document=to_document_response(document), duplicate=duplicate)


@router.patch("/documents/{document_id}", response_model=DocumentResponse)
def update_document_metadata(
    document_id: str,
    payload: DocumentMetadataUpdate,
    container: ServiceContainer = Depends(get_container),
):
    try:
        document = container.document_service.update_metadata(
            document_id,
            brand=payload.brand,
            model=payload.model,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="文档不存在。") from exc
    except DocumentUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return to_document_response(document)

# 根据文档 ID，启动这个 PDF文档的索引任务。
@router.post(
    "/documents/{document_id}/index",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def start_indexing(
    document_id: str,
    background_tasks: BackgroundTasks,
    container: ServiceContainer = Depends(get_container),
):
    # 后台任务执行器
    executor = FastAPIBackgroundTaskExecutor(
        background_tasks,
        container.indexing.execute,
    )
    try:
        document = container.indexing.submit_index(document_id, executor)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="文档不存在。") from exc
    except DocumentStateConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except IndexTaskSubmissionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return to_document_response(document)

# 根据ID删除文档
@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    container: ServiceContainer = Depends(get_container),
) -> None:
    try:
        container.indexing.delete_document(document_id)
    except DocumentStateConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DocumentDeletionError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

# 用户提问解答流程
@router.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    container: ServiceContainer = Depends(get_container),
):
    session_id = str(payload.session_id or uuid4())
    try:
        result = container.conversation.answer(session_id, payload.query, payload.model)
    except Exception as exc:
        logger.exception("chat failed")
        raise HTTPException(
            status_code=503,
            detail="检索或模型服务暂时不可用，请检查 API 配置和网络连接。",
        ) from exc
    try:
        sources = build_cited_source_responses(result)
    except ValueError as exc:
        logger.error("回答来源映射不一致：%s", exc)
        return ChatResponse(
            answer="回答来源校验异常，暂时无法提供可靠答案，请稍后重试或联系石头官方人工客服。",
            refused=True,
            refusal_code=RefusalCode.INVALID_CITATION,
            refusal_reason="回答引用与后端可信来源映射不一致。",
            sources=[],
            session_id=session_id,
            standalone_query=result.get("standalone_query", payload.query),
            history_used=bool(result.get("history_used", False)),
        )
    return ChatResponse(
        answer=result.get("answer", ""),
        refused=bool(result.get("refused", False)), # 是否拒绝
        refusal_code=result.get("refusal_code"),
        refusal_reason=result.get("refusal_reason"), # 拒绝的原因，没有拒绝时值为空。
        sources=sources, # 回答的来源
        session_id=session_id,
        standalone_query=result.get("standalone_query", payload.query),
        history_used=bool(result.get("history_used", False)),
    )

