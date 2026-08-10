from pydantic import BaseModel, Field

class AssistantResponse(BaseModel):
    """Schema for forcing structured outputs from the language model."""
    answer: str = Field(
        description="The detailed answer to the question based strictly on context."
    )
    confidence: str = Field(
        description="Confidence level in the response: 'High', 'Medium', or 'Low'."
    )
    sources_used: bool = Field(
        description="True if context provided relevant data; False if context was insufficient."
    )