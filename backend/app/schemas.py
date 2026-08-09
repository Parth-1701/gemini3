from pydantic import BaseModel, Field


class LabEvent(BaseModel):
    timestamp_seconds: float = Field(
        description="Timestamp in seconds from the start of the video."
    )
    event_description: str = Field(
        description="Description of what happens at this timestamp."
    )
    safety_violation: bool = Field(
        description="True if this moment includes a safety protocol violation."
    )


class LabReport(BaseModel):
    summary: str = Field(
        description="Short summary of the experiment and overall observations."
    )
    events: list[LabEvent] = Field(
        description="Chronological list of lab events detected in the video."
    )


class UploadResponse(BaseModel):
    name: str
    file_uri: str
    mime_type: str


class AnalyzeRequest(BaseModel):
    file_uri: str
    mime_type: str = "video/mp4"
