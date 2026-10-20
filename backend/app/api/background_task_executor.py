from collections.abc import Callable

from fastapi import BackgroundTasks


class FastAPIBackgroundTaskExecutor:
    """使用 FastAPI 进程内后台任务执行索引。"""

    def __init__(
        self,
        background_tasks: BackgroundTasks,
        task: Callable[[str], None],
    ):
        self._background_tasks = background_tasks
        self._task = task

    def submit(self, document_id: str) -> None:
        self._background_tasks.add_task(self._task, document_id)
