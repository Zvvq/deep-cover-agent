import uvicorn

from .config import get_settings


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "deep_cover_agent.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    run()
