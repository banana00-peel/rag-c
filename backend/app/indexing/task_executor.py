from typing import Protocol


class IndexTaskExecutor(Protocol):
    """索引任务提交边界，后续可以替换为 Celery 等外部任务队列。"""

    def submit(self, document_id: str) -> None:
        """提交一个已经原子抢占执行权的文档索引任务。"""

