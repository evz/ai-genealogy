"""Client for the Loghi HTR orchestrator (webservice/orchestrator) HTTP API"""

import logging

import requests
from django.conf import settings

logger = logging.getLogger(__name__)


class LoghiClient:
    """Client for the Loghi handwritten text recognition orchestrator"""

    def __init__(self, host: str | None = None, port: int | None = None, timeout: int = 600):
        self.host = host or settings.LOGHI_HOST
        self.port = port or settings.LOGHI_PORT
        self.base_url = f"http://{self.host}:{self.port}"
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if the Loghi orchestrator and its dependent services are healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def transcribe(self, image_path: str) -> str:
        """
        Submit an image to the Loghi orchestrator for end-to-end handwritten text recognition.

        Args:
            image_path: Path to the image file on disk

        Returns:
            Plain-text transcription

        Raises:
            RuntimeError: If the orchestrator returns a non-200 response
        """
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {"output_format": "text"}
            response = requests.post(
                f"{self.base_url}/transcribe",
                files=files,
                data=data,
                timeout=self.timeout,
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Loghi transcription failed: {response.status_code} - {response.text}"
            )

        return response.text
