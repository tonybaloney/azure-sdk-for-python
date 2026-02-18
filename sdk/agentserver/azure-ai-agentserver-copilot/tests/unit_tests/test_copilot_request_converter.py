import pytest
from azure.ai.agentserver.core import models

from azure.ai.agentserver.copilot.models import CopilotRequestConverter


@pytest.mark.unit
def test_convert_string_input():
    """Test that a plain string input is returned as-is."""
    create_response = models.CreateResponse(input="hello world")
    converter = CopilotRequestConverter(create_response)
    result = converter.convert()
    assert result == "hello world"


@pytest.mark.unit
def test_convert_none_input():
    """Test that None input returns an empty string."""
    create_response = models.CreateResponse(input=None)
    converter = CopilotRequestConverter(create_response)
    result = converter.convert()
    assert result == ""


@pytest.mark.unit
def test_convert_implicit_user_message():
    """Test conversion of implicit user message dict with content key."""
    create_response = models.CreateResponse(
        input=[{"content": "What is the weather?"}],
    )
    converter = CopilotRequestConverter(create_response)
    result = converter.convert()
    assert result == "What is the weather?"


@pytest.mark.unit
def test_convert_multiple_messages():
    """Test conversion of multiple message dicts into a joined prompt."""
    create_response = models.CreateResponse(
        input=[
            {"content": "You are a helpful assistant."},
            {"content": "What is 2+2?"},
        ],
    )
    converter = CopilotRequestConverter(create_response)
    result = converter.convert()
    assert "You are a helpful assistant." in result
    assert "What is 2+2?" in result


@pytest.mark.unit
def test_convert_message_with_content_parts():
    """Test conversion of a message with structured content parts."""
    create_response = models.CreateResponse(
        input=[
            {
                "content": [
                    {"type": "input_text", "text": "Hello"},
                    {"type": "input_text", "text": "World"},
                ]
            }
        ],
    )
    converter = CopilotRequestConverter(create_response)
    result = converter.convert()
    assert "Hello" in result
    assert "World" in result


@pytest.mark.unit
def test_convert_empty_message_list():
    """Test conversion of an empty message list returns empty string."""
    create_response = models.CreateResponse(input=[])
    converter = CopilotRequestConverter(create_response)
    result = converter.convert()
    assert result == ""


@pytest.mark.unit
def test_convert_unsupported_input_raises():
    """Test that an unsupported input type raises ValueError."""
    create_response = models.CreateResponse(input=12345)
    converter = CopilotRequestConverter(create_response)
    with pytest.raises(ValueError, match="Unsupported input type"):
        converter.convert()
