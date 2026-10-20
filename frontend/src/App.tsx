import { useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  Bot,
  CheckCircle2,
  ChevronDown,
  Database,
  FileText,
  LoaderCircle,
  MessageSquarePlus,
  RotateCcw,
  RefreshCw,
  Send,
  Trash2,
  UploadCloud,
  UserRound,
} from "lucide-react";

import {
  askQuestion,
  deleteDocument,
  listDocuments,
  startIndexing,
  updateDocument,
  uploadDocument,
} from "./api";
import type { ChatResponse, DocumentRecord, SourceRecord } from "./types";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  response?: ChatResponse;
};

const MODELS = ["all", "G20", "G10S Pro", "S5"];

function statusLabel(status: DocumentRecord["status"]): string {
  return {
    pending: "待确认",
    processing: "索引中",
    completed: "已完成",
    failed: "失败",
  }[status];
}

function pageLabel(pageStart: number, pageEnd: number): string {
  return pageStart === pageEnd ? `第 ${pageStart} 页` : `第 ${pageStart}-${pageEnd} 页`;
}

function scoreLabel(score: number | null): string {
  return score === null ? "--" : score.toFixed(4);
}

function sourceElementId(messageId: string, sourceId: string): string {
  return `source-${messageId}-${sourceId}`;
}

type CitationTextProps = {
  content: string;
  sources: SourceRecord[];
  onCitationClick: (sourceId: string) => void;
};

function CitationText({ content, sources, onCitationClick }: CitationTextProps) {
  const availableSourceIds = new Set(sources.map((source) => source.source_id));
  const fragments = content.split(/(\[S[1-9]\d*\])/g);

  return fragments.map((fragment, index) => {
    const match = fragment.match(/^\[(S[1-9]\d*)\]$/);
    const sourceId = match?.[1];
    if (!sourceId || !availableSourceIds.has(sourceId)) {
      return fragment;
    }
    return (
      <button
        className="citation-button"
        key={`${sourceId}-${index}`}
        type="button"
        title={`查看来源 ${sourceId}`}
        onClick={() => onCitationClick(sourceId)}
      >
        [{sourceId}]
      </button>
    );
  });
}

function App() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<DocumentRecord | null>(null);
  const [brand, setBrand] = useState("石头");
  const [model, setModel] = useState("");
  const [filterModel, setFilterModel] = useState("all");
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [savingMetadata, setSavingMetadata] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [highlightedSource, setHighlightedSource] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [sessionId, setSessionId] = useState<string>(() => crypto.randomUUID());

  const refreshDocuments = async () => {
    try {
      setDocuments(await listDocuments());
    } catch (err) {
      setError(err instanceof Error ? err.message : "无法加载文档列表。");
    }
  };

  useEffect(() => {
    void refreshDocuments();
  }, []);

  useEffect(() => {
    const hasProcessing = documents.some((document) => document.status === "processing");
    if (!hasProcessing) {
      return;
    }
    const timer = window.setInterval(() => {
      void refreshDocuments();
    }, 1500);
    return () => window.clearInterval(timer);
  }, [documents]);

  useEffect(() => {
    if (!selectedDocument) {
      return;
    }
    const refreshed = documents.find((document) => document.id === selectedDocument.id);
    if (refreshed) {
      setSelectedDocument(refreshed);
      setBrand(refreshed.brand);
      setModel(refreshed.model);
    }
  }, [documents]);

  const selectDocument = (document: DocumentRecord) => {
    setSelectedDocument(document);
    setBrand(document.brand);
    setModel(document.model);
    setError("");
    setNotice("");
  };

  const handleUpload = async (file: File) => {
    setUploading(true);
    setError("");
    setNotice("");
    try {
      const result = await uploadDocument(file);
      selectDocument(result.document);
      setNotice(result.duplicate ? "该文件已存在，已定位到原文档。" : "文件已上传，请确认型号后开始索引。");
      await refreshDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "上传失败。");
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleSaveAndIndex = async () => {
    if (!selectedDocument || !brand.trim() || !model.trim()) {
      setError("请填写品牌和型号。");
      return;
    }
    setSavingMetadata(true);
    setError("");
    try {
      let target = selectedDocument;
      if (selectedDocument.status !== "completed") {
        target = await updateDocument(selectedDocument.id, {
          brand: brand.trim(),
          model: model.trim(),
        });
      }
      const indexingDocument = await startIndexing(target.id);
      setSelectedDocument(indexingDocument);
      setNotice(
        selectedDocument.status === "completed" ? "已提交重新索引任务。" : "已提交索引任务。",
      );
      await refreshDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "索引任务提交失败。");
    } finally {
      setSavingMetadata(false);
    }
  };

  const handleDelete = async () => {
    if (!selectedDocument || selectedDocument.status === "processing") {
      return;
    }
    const confirmed = window.confirm(`确定删除“${selectedDocument.filename}”及其索引数据吗？`);
    if (!confirmed) {
      return;
    }

    setSavingMetadata(true);
    setError("");
    setNotice("");
    try {
      await deleteDocument(selectedDocument.id);
      setSelectedDocument(null);
      setBrand("石头");
      setModel("");
      setNotice("文档及其索引数据已删除。");
      await refreshDocuments();
    } catch (err) {
      setError(err instanceof Error ? err.message : "文档删除失败。");
    } finally {
      setSavingMetadata(false);
    }
  };

  const handleAsk = async () => {
    const trimmedQuery = query.trim();
    if (!trimmedQuery || loading) {
      return;
    }
    setLoading(true);
    setError("");
    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmedQuery,
    };
    setMessages((current) => [...current, userMessage]);
    setQuery("");

    try {
      const response = await askQuestion(trimmedQuery, filterModel, sessionId);
      setSessionId(response.session_id);
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: response.answer,
          response,
        },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "回答生成失败。");
    } finally {
      setLoading(false);
    }
  };

  const handleNewConversation = () => {
    if (loading) {
      return;
    }
    if (messages.length > 0 && !window.confirm("确定开始新会话并清空当前聊天记录吗？")) {
      return;
    }
    setSessionId(crypto.randomUUID());
    setMessages([]);
    setQuery("");
    setError("");
    setNotice("");
    setHighlightedSource(null);
  };

  const onQueryKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleAsk();
    }
  };

  const handleCitationClick = (messageId: string, sourceId: string) => {
    const targetId = sourceElementId(messageId, sourceId);
    const target = document.getElementById(targetId);
    if (!(target instanceof HTMLDetailsElement)) {
      return;
    }
    target.open = true;
    setHighlightedSource(targetId);
    target.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark">R</div>
          <div>
            <p className="eyebrow">KNOWLEDGE ASSISTANT</p>
            <h1>Roborock RAG</h1>
          </div>
        </div>
        <div className="system-status">
          <span className="status-dot" />
          <span>本地知识库</span>
        </div>
      </header>

      <main className="workspace">
        <aside className="sidebar">
          <section className="panel knowledge-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">KNOWLEDGE BASE</p>
                <h2>官方说明书</h2>
              </div>
              <Database size={18} strokeWidth={1.8} />
            </div>

            <button
              className="upload-zone"
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
            >
              {uploading ? (
                <LoaderCircle className="spin" size={24} />
              ) : (
                <UploadCloud size={24} />
              )}
              <span>{uploading ? "正在上传" : "上传 PDF 说明书"}</span>
              <small>支持增量索引</small>
            </button>
            <input
              ref={fileInputRef}
              className="hidden-input"
              type="file"
              accept="application/pdf,.pdf"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) {
                  void handleUpload(file);
                }
              }}
            />

            <div className="document-list">
              <div className="list-label">
                <span>已导入文档</span>
                <button
                  className="icon-button"
                  title="刷新文档列表"
                  type="button"
                  onClick={() => void refreshDocuments()}
                >
                  <RefreshCw size={15} />
                </button>
              </div>
              {documents.length === 0 ? (
                <div className="empty-state">
                  <FileText size={20} />
                  <span>暂无文档</span>
                </div>
              ) : (
                documents.map((document) => (
                  <button
                    className={`document-item ${selectedDocument?.id === document.id ? "selected" : ""}`}
                    key={document.id}
                    type="button"
                    onClick={() => selectDocument(document)}
                  >
                    <FileText size={17} />
                    <span className="document-copy">
                      <strong>{document.filename}</strong>
                      <small>
                        {document.model || "未识别型号"} · {statusLabel(document.status)}
                      </small>
                    </span>
                    {document.status === "completed" && <CheckCircle2 size={16} />}
                    {document.status === "processing" && <LoaderCircle className="spin" size={16} />}
                    {document.status === "failed" && <AlertCircle size={16} />}
                  </button>
                ))
              )}
            </div>
          </section>

          {selectedDocument && (
            <section className="panel metadata-panel">
              <div className="panel-heading compact">
                <div>
                  <p className="eyebrow">DOCUMENT METADATA</p>
                  <h2>确认文档信息</h2>
                </div>
              </div>
              <label>
                品牌
                <input
                  value={brand}
                  disabled={
                    selectedDocument.status === "processing" || selectedDocument.status === "completed"
                  }
                  onChange={(event) => setBrand(event.target.value)}
                />
              </label>
              <label>
                型号
                <input
                  value={model}
                  placeholder="例如：G20"
                  disabled={
                    selectedDocument.status === "processing" || selectedDocument.status === "completed"
                  }
                  onChange={(event) => setModel(event.target.value)}
                />
              </label>
              <button
                className="primary-button"
                type="button"
                onClick={() => void handleSaveAndIndex()}
                disabled={
                  savingMetadata ||
                  selectedDocument.status === "processing"
                }
              >
                {savingMetadata ? (
                  <LoaderCircle className="spin" size={16} />
                ) : selectedDocument.status === "completed" ? (
                  <RotateCcw size={16} />
                ) : (
                  <Database size={16} />
                )}
                {selectedDocument.status === "completed"
                  ? "重新索引"
                  : selectedDocument.status === "failed"
                    ? "保存并重试"
                    : "保存并开始索引"}
              </button>
              <button
                className="danger-button"
                type="button"
                onClick={() => void handleDelete()}
                disabled={savingMetadata || selectedDocument.status === "processing"}
              >
                <Trash2 size={16} />
                删除文档
              </button>
              {selectedDocument.error_message && (
                <p className="inline-error">{selectedDocument.error_message}</p>
              )}
            </section>
          )}
        </aside>

        <section className="chat-panel">
          <div className="chat-heading">
            <div>
              <p className="eyebrow">ROBOROCK SUPPORT</p>
              <h2>智能客服工作台</h2>
              <p className="heading-note">基于已索引的官方中文说明书回答问题</p>
            </div>
            <div className="chat-actions">
              <label className="model-select">
                <span>检索型号</span>
                <div>
                  <select
                    value={filterModel}
                    onChange={(event) => setFilterModel(event.target.value)}
                  >
                    {MODELS.map((item) => (
                      <option key={item} value={item}>
                        {item === "all" ? "全部型号" : item}
                      </option>
                    ))}
                  </select>
                  <ChevronDown size={15} />
                </div>
              </label>
              <button
                className="new-conversation-button"
                type="button"
                onClick={handleNewConversation}
                disabled={loading}
              >
                <MessageSquarePlus size={15} />
                新建会话
              </button>
            </div>
          </div>

          <div className="conversation">
            {messages.length === 0 ? (
              <div className="welcome-state">
                <div className="welcome-icon">
                  <Bot size={28} strokeWidth={1.6} />
                </div>
                <h3>开始查询说明书</h3>
                <p>输入清洁、配网、故障或维护相关问题。</p>
              </div>
            ) : (
              messages.map((message) => (
                <article className={`message-row ${message.role}`} key={message.id}>
                  <div className="message-avatar">
                    {message.role === "user" ? <UserRound size={16} /> : <Bot size={17} />}
                  </div>
                  <div className="message-body">
                    <div className="message-role">{message.role === "user" ? "你" : "Roborock Assistant"}</div>
                    <div className="message-content">
                      <CitationText
                        content={message.content}
                        sources={message.response?.sources ?? []}
                        onCitationClick={(sourceId) => handleCitationClick(message.id, sourceId)}
                      />
                    </div>
                    {message.response && message.response.sources.length > 0 && (
                      <div className="sources">
                        <div className="sources-title">来源</div>
                        {message.response.sources.map((source) => {
                          const elementId = sourceElementId(message.id, source.source_id);
                          return (
                            <details
                              className={`source-item ${highlightedSource === elementId ? "highlighted" : ""}`}
                              id={elementId}
                              key={source.source_id}
                            >
                              <summary className="source-summary">
                                <FileText size={14} />
                                <span className="source-id">{source.source_id}</span>
                                <span className="source-meta">
                                  {source.document_name} · {pageLabel(source.page_start, source.page_end)} · {source.model}
                                </span>
                                <ChevronDown className="source-chevron" size={14} />
                              </summary>
                              <div className="source-content">
                                <div className="snippet-label">原文片段</div>
                                <p className="source-snippet">{source.snippet}</p>
                                <div className="score-grid">
                                  <span>Dense {scoreLabel(source.dense_score)}</span>
                                  <span>BM25 {scoreLabel(source.bm25_score)}</span>
                                  <span>RRF {scoreLabel(source.rrf_score)}</span>
                                </div>
                                {source.dense_distance !== null && (
                                  <small>Dense distance {source.dense_distance.toFixed(4)}</small>
                                )}
                              </div>
                            </details>
                          );
                        })}
                      </div>
                    )}
                    {message.response?.refused && message.response.refusal_reason && (
                      <div className="refusal-reason">
                        <AlertCircle size={13} />
                        <span>{message.response.refusal_reason}</span>
                      </div>
                    )}
                  </div>
                </article>
              ))
            )}
            {loading && (
              <div className="message-row assistant">
                <div className="message-avatar">
                  <Bot size={17} />
                </div>
                <div className="message-body">
                  <div className="message-role">Roborock Assistant</div>
                  <div className="typing-state">
                    <span />
                    <span />
                    <span />
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="composer-wrap">
            {error && (
              <div className="banner error-banner">
                <AlertCircle size={16} />
                <span>{error}</span>
              </div>
            )}
            {notice && (
              <div className="banner notice-banner">
                <CheckCircle2 size={16} />
                <span>{notice}</span>
              </div>
            )}
            <div className="composer">
              <textarea
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={onQueryKeyDown}
                placeholder="例如：G20 如何清理尘盒？"
                rows={2}
              />
              <button
                className="send-button"
                title="发送问题"
                type="button"
                onClick={() => void handleAsk()}
                disabled={loading || !query.trim()}
              >
                {loading ? <LoaderCircle className="spin" size={18} /> : <Send size={18} />}
              </button>
            </div>
            <div className="composer-footer">
              <span>Enter 发送 · Shift + Enter 换行</span>
              <span>低置信度问题将转人工客服</span>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;

