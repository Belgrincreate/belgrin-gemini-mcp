#!/usr/bin/env python3
"""
Gemini MCP Server for Belgrin.
Hosted on Render -- provides image generation and text generation
via Google's Gemini API for use inside Cowork sessions.
"""

import os
import json
import httpx
import uvicorn
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import TransportSecuritySettings

mcp = FastMCP(
    "gemini_mcp",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False)
)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
API_BASE = "https://generativelanguage.googleapis.com/v1beta"
# Override image model via env var if needed: set IMAGE_GENERATION_MODEL on Render
IMAGE_GENERATION_MODEL = os.environ.get("IMAGE_GENERATION_MODEL", "gemini-2.0-flash-exp")


def _handle_error(e: Exception) -> str:
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        if status == 400:
            return f"Error: Bad request — check your prompt. Details: {e.response.text}"
        elif status == 403:
            return "Error: API key rejected. Check your GEMINI_API_KEY environment variable."
        elif status == 429:
            return "Error: Rate limit reached. Wait 30 seconds and try again."
        return f"Error: API returned status {status}. Details: {e.response.text}"
    elif isinstance(e, httpx.TimeoutException):
        return "Error: Request timed out — image generation can take up to 30 seconds. Try again."
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


class GenerateTextInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    prompt: str = Field(..., description="The text prompt.", min_length=1, max_length=10000)
    temperature: Optional[float] = Field(default=0.7, description="Creativity 0.0-1.0", ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=1024, description="Max output tokens", ge=1, le=8192)


class ListModelsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filter: Optional[str] = Field(default=None, description="Optional filter string e.g. 'imagen' or 'flash'")


@mcp.tool(
    name="gemini_list_models",
    annotations={
        "title": "List Available Gemini Models",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def gemini_list_models(params: ListModelsInput) -> str:
    """List all models available with the configured Gemini API key."""
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY not configured on the server."

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{API_BASE}/models",
                params={"key": GEMINI_API_KEY},
            )
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])

            if params.filter:
                models = [m for m in models if params.filter.lower() in m.get("name", "").lower()]

            result = []
            for m in models:
                result.append({
                    "name": m.get("name", ""),
                    "displayName": m.get("displayName", ""),
                    "supportedMethods": m.get("supportedGenerationMethods", []),
                })

            return json.dumps({"total": len(result), "models": result}, indent=2)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="gemini_generate_image",
    annotations={
        "title": "Generate Image with Gemini",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def gemini_generate_image(params: GenerateImageInput) -> str:
    """Generate an image from a text prompt using Gemini's image generation model."""
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY not configured on the server. Contact Michael."

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                f"{API_BASE}/models/{IMAGE_GENERATION_MODEL}:generateContent",
                params={"key": GEMINI_API_KEY},
                json={
                    "contents": [{"parts": [{"text": params.prompt}]}],
                    "generationConfig": {
                        "responseModalities": ["IMAGE", "TEXT"],
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

            images = []
            for candidate in data.get("candidates", []):
                for part in candidate.get("content", {}).get("parts", []):
                    if "inlineData" in part:
                        images.append({
                            "mime_type": part["inlineData"].get("mimeType", "image/png"),
                            "b64_json": part["inlineData"].get("data", ""),
                        })

            if not images:
                return f"No images returned. Raw response: {json.dumps(data)}"

            result = {
                "success": True,
                "prompt": params.prompt,
                "model_used": IMAGE_GENERATION_MODEL,
                "images_generated": len(images),
                "images": [{"index": i, **img} for i, img in enumerate(images)],
            }
            return json.dumps(result)
    except Exception as e:
        return _handle_error(e)


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
    """Generate text using Gemini 2.0 Flash."""
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY not configured on the server. Contact Michael."

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{API_BASE}/models/gemini-2.0-flash:generateContent",
                params={"key": GEMINI_API_KEY},
                json={
                    "contents": [{"parts": [{"text": params.prompt}]}],
                    "generationConfig": {
                        "temperature": params.temperature,
                        "maxOutputTokens": params.max_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return "No text generated."
            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            return json.dumps({"success": True, "text": text})
    except Exception as e:
        return _handle_error(e)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app = mcp.streamable_http_app()
    uvicorn.run(app, host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")

