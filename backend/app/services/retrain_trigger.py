import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


async def trigger_github_retrain(drift_score: float) -> dict:
    if not settings.github_repo or not settings.github_token:
        logger.warning("GitHub retrain skipped: GITHUB_REPO or GITHUB_TOKEN not configured")
        return {
            "triggered": False,
            "reason": "github_not_configured",
            "message": "Set GITHUB_REPO and GITHUB_TOKEN to enable automated retraining via GitHub Actions",
        }

    url = f"https://api.github.com/repos/{settings.github_repo}/actions/workflows/{settings.github_workflow_id}/dispatches"

    payload = {
        "ref": "main",
        "inputs": {
            "drift_score": str(drift_score),
            "trigger_source": "evidently-drift-monitor",
        },
    }

    headers = {
        "Authorization": f"Bearer {settings.github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload, headers=headers)

    if response.status_code == 204:
        return {"triggered": True, "status_code": 204}

    return {
        "triggered": False,
        "status_code": response.status_code,
        "body": response.text,
    }
