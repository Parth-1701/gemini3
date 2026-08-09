import os
import re
import json
import time

from fastapi import HTTPException
from google import genai
from google.genai import types

from .schemas import LabReport


_client = None


LAB_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string"
        },
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp_seconds": {
                        "type": "number"
                    },
                    "event_description": {
                        "type": "string"
                    },
                    "safety_violation": {
                        "type": "boolean"
                    }
                },
                "required": [
                    "timestamp_seconds",
                    "event_description",
                    "safety_violation"
                ]
            }
        }
    },
    "required": [
        "summary",
        "events"
    ]
}


PROMPT = """
You are an expert chemistry professor grading a student's lab experiment.

Watch the attached video carefully.

Identify every time:
1. A new chemical is introduced.
2. A physical or chemical reaction occurs.
3. A notable procedural step happens.
4. A safety protocol is violated.

For each event, provide:
- The timestamp in seconds.
- A clear description of what happens.
- Whether this moment includes a safety violation.

Return the data strictly matching the requested JSON schema.
Do not include markdown.
Do not include explanations outside the JSON.
"""


def get_client():
    global _client

    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise HTTPException(
                status_code=500,
                detail="GEMINI_API_KEY is not set."
            )

        _client = genai.Client(api_key=api_key)

    return _client


def _get_file_name(obj):
    name = getattr(obj, "name", None)

    if name:
        return name

    nested_file = getattr(obj, "file", None)

    if nested_file:
        return getattr(nested_file, "name", None)

    return None


def _get_file_uri(obj):
    uri = getattr(obj, "uri", None)

    if uri:
        return uri

    nested_file = getattr(obj, "file", None)

    if nested_file:
        return getattr(nested_file, "uri", None)

    return None


def _get_state_name(file_obj):
    state = getattr(file_obj, "state", "")

    if hasattr(state, "name"):
        state = state.name

    return str(state).upper()


def _wait_for_file(file_name: str, timeout_seconds: int = 240):
    client = get_client()
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        file_obj = client.files.get(name=file_name)
        state = _get_state_name(file_obj)

        if "ACTIVE" in state:
            return file_obj

        if "FAILED" in state:
            raise HTTPException(
                status_code=500,
                detail="Uploaded video failed processing in Gemini."
            )

        time.sleep(2)

    raise HTTPException(
        status_code=504,
        detail="Timed out waiting for Gemini to process the uploaded video."
    )


def upload_file_to_gemini(local_path: str, display_name: str, mime_type: str):
    client = get_client()

    try:
        upload_config = types.UploadFileConfig(
            display_name=display_name,
            mime_type=mime_type
        )
        uploaded = client.files.upload(
            file=local_path,
            config=upload_config
        )
    except AttributeError:
        uploaded = client.files.upload(
            file=local_path,
            display_name=display_name,
            mime_type=mime_type
        )

    file_name = _get_file_name(uploaded)

    if not file_name:
        raise HTTPException(
            status_code=500,
            detail="Gemini File API did not return a file name."
        )

    active_file = _wait_for_file(file_name)

    file_uri = _get_file_uri(active_file)

    if not file_uri:
        raise HTTPException(
            status_code=500,
            detail="Gemini File API did not return a file URI."
        )

    returned_mime_type = getattr(active_file, "mime_type", None) or mime_type

    return {
        "name": file_name,
        "file_uri": file_uri,
        "mime_type": returned_mime_type
    }


def _parse_json_response(response):
    parsed = getattr(response, "parsed", None)

    if parsed is not None:
        if hasattr(parsed, "model_dump"):
            return parsed.model_dump()

        if isinstance(parsed, dict):
            return parsed

        if isinstance(parsed, str):
            return json.loads(parsed)

        return parsed

    text = getattr(response, "text", "") or ""
    text = text.strip()

    # Remove markdown code fences if Gemini accidentally returns them.
    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text,
        flags=re.MULTILINE
    )

    # Extract the first JSON object if there is extra text around it.
    start = text.find("{")
    end = text.rfind("}") + 1

    if start != -1 and end > start:
        text = text[start:end]

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini returned invalid JSON. Raw response begins: {text[:1000]}"
        ) from exc


def generate_lab_report(file_uri: str, mime_type: str):
    client = get_client()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    try:
        video_part = types.Part.from_uri(
            file_uri=file_uri,
            mime_type=mime_type
        )
    except Exception:
        video_part = {
            "file_data": {
                "file_uri": file_uri,
                "mime_type": mime_type
            }
        }

    try:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=LAB_REPORT_SCHEMA,
            temperature=0.1
        )
    except Exception:
        config = {
            "response_mime_type": "application/json",
            "response_schema": LAB_REPORT_SCHEMA,
            "temperature": 0.1
        }

    response = client.models.generate_content(
        model=model,
        contents=[
            video_part,
            PROMPT
        ],
        config=config
    )

    data = _parse_json_response(response)

    return LabReport.model_validate(data)
