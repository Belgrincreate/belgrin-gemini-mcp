#!/usr/bin/env python3
"""
Gemini MCP Server for Belgrin.
Hosted on Render -- provides image generation and text generation
via Google's Gemini API for use inside Cowork sessions.
"""

import os
import json
import httpx
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("gemini_mcp")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _handle_error(e: Exception) -> str:
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        if status == 400:
            return f"Error: Bad request -- check your prompt. Details: {e.response.text}"
        elif status == 403:
            return "Error: API key rejected. Check your GEMINI_API_KEY environment variable."
        elif status == 429:
            return "Error: Rate limit reached. Wait 30 seconds and try again."
        return f"Error: API returned status {status}. Details: {e.response.text}"
    elif isinstance(e, httpx.TimeoutException):
        return "Error: Request timed out -- image generation can take up to 30 seconds. Try again."
    return f"Error: {type(e).__name__}: {str(e)}"


class GenerateImageInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    prompt: str = Field(
        ...,
        description="Detailed description of the image.",
        min_length=10,
        max_length=2000,
    )
    aspect_ratio: Optional[str] = Field(
        default="1:1",
        description="Shape of the image: '1:1' square, '9:16' portrait/stories, '16:9' landscape, '4:3' standard",
    )
    num_images: Optional[int] = Field(
        default=1,
        description="Number of image variations to generate (1-4)",
        ge=1,
        le=4,
    )


@mcp.tool(
    name="gemini_generate_image",
    annotations={
        "title": "Generate Image with Gemini Imagen",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def gemini_generate_image(params: GenerateImageInput) -> str:
    """Generate an image from a text prompt using Google's Imagen 3 model."""
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY not configured on the server. Contact Michael."

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                f"{API_BASE}/models/imagen-3.0-generate-002:predict",
                params={"key": GEMINI_API_KEY},
                json={
                    "instances": [{"prompt": params.prompt}],
                    "parameters": {
                        "sampleCount": params.num_images,
                        "aspectRatio": params.aspect_ratio,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            predictions = data.get("predictions", [])
            if not predictions:
                return "No images returned. Try a more descriptive prompt."
            result = {
                "success": True,
                "prompt": params.prompt,
                "aspect_ratio": params.aspect_ratio,
                "images_generated": len(predictions),
                "images": [
                    {
                        "index": i + 1,
                        "mime_type": pred.get("mimeType", "image/png"),
                        "base64": pred.get("bytesBase64Encoded", ""),
                    }
                    for i, pred in enumerate(predictions)
                ],
            }
            return json.dumps(result, indent=2)
    except Exception as e:
        return _handle_error(e)


class GenerateTextInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")
    prompt: str = Field(
        ...,
        description="Your instruction.",
        min_length=5,
        max_length=10000,
    )
    temperature: Optional[float] = Field(
        default=0.7,
        description="Creativity: 0.0 = precise, 1.0 = very creative",
        ge=0.0,
        le=1.0,
    )


@mcp.tool(
    name="gemini_generate_text",
    annotations={
        "title": "Generate Text with Gemini",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def gemini_generate_text(params: GenerateTextInput) -> str:
    """Generate text using Google's Gemini 2.0 Flash model."""
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY not configured on the server."

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{API_BASE}/models/gemini-2.0-flash:generateContent",
                params={"key": GEMINI_API_KEY},
                json={
                    "contents": [{"parts": [{"text": params.prompt}]}],
                    "generationConfig": {"temperature": params.temperature},
                },
            )
            response.raise_for_status()
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return "No response generated. Try rephrasing your prompt."
            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return text or "Empty response. Try again."
    except Exception as e:
        return _handle_error(e)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
