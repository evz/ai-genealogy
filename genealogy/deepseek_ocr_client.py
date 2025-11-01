#!/usr/bin/env python3
"""
DeepSeek-OCR Client

Lightweight client for communicating with DeepSeek-OCR server via ZeroMQ.
"""

import base64
import io
import json
import logging

import zmq
from PIL import Image

logger = logging.getLogger(__name__)


class DeepSeekOCRClient:
    """Client for remote DeepSeek-OCR inference via ZeroMQ"""

    def __init__(self, host="localhost", port=5555, timeout=300000):
        """
        Initialize connection to DeepSeek-OCR server

        Args:
            host: Server hostname or IP
            port: Server port
            timeout: Request timeout in milliseconds (default: 5 minutes for gundam mode)
        """
        self.host = host
        self.port = port
        self.timeout = timeout

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout)
        self.socket.connect(f"tcp://{host}:{port}")

        logger.info(f"Connected to DeepSeek-OCR server at {host}:{port}")

    def process_image(self, image, mode="gundam", preserve_layout=True):
        """
        Process an image with DeepSeek-OCR

        Args:
            image: PIL Image
            mode: Resolution mode ('tiny', 'small', 'base', 'large', 'gundam')
            preserve_layout: Whether to include grounding tokens

        Returns:
            Dictionary with 'text', 'layout', and 'metadata'
        """
        # Convert to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()

        # Encode as base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        # Build request
        request = {
            'image': image_b64,
            'mode': mode,
            'preserve_layout': preserve_layout
        }

        try:
            # Send and receive
            self.socket.send_string(json.dumps(request))
            response_str = self.socket.recv_string()
            response = json.loads(response_str)

            # Check for errors
            if 'error' in response:
                raise RuntimeError(f"Server error: {response['error']}")

            return response

        except zmq.Again:
            logger.error(f"Request timeout after {self.timeout}ms")
            raise TimeoutError(f"Server did not respond within {self.timeout}ms")

        except zmq.ZMQError as e:
            logger.error(f"ZMQ error: {e}")
            raise ConnectionError(f"Failed to communicate with server: {e}")

    def close(self):
        """Close connection to server"""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        logger.info("Closed connection to DeepSeek-OCR server")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
