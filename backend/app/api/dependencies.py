from fastapi import Request

from app.core.container import ServiceContainer


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container

