import base64
import os

import pytest
from azure.ai.agentserver.core import models

from azure.ai.agentserver.copilot.models import ConvertedAttachments, CopilotRequestConverter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _b64(text: str) -> str:
    """Return *text* encoded as a base64 ASCII string (simulate file_data)."""
    return base64.b64encode(text.encode()).decode()


def _data_uri(mime: str, text: str) -> str:
    """Return a ``data:<mime>;base64,<data>`` URI for *text*."""
    return f"data:{mime};base64,{_b64(text)}"


def _make_converter(input_value) -> CopilotRequestConverter:
    return CopilotRequestConverter(models.CreateResponse(input=input_value))



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


# ===========================================================================
# Existing convert() behaviour — non-text content parts
# ===========================================================================


@pytest.mark.unit
class TestConvertContentAnnotations:
    """convert() should annotate non-materialisable attachments as text hints."""

    def test_external_image_url_annotated_in_prompt(self):
        url = "https://example.com/photo.jpg"
        cr = _make_converter(
            [{"content": [{"type": "input_image", "image_url": {"url": url}}]}]
        )
        result = cr.convert()
        assert f"[image: {url}]" in result

    def test_data_uri_image_not_annotated_in_prompt(self):
        """Base64 data images are materialised as attachments, not annotated."""
        uri = _data_uri("image/png", "fake-png-bytes")
        cr = _make_converter(
            [{"content": [{"type": "input_image", "image_url": {"url": uri}}]}]
        )
        result = cr.convert()
        assert "[image:" not in result

    def test_image_file_id_annotated_in_prompt(self):
        cr = _make_converter(
            [{"content": [{"type": "input_image", "file_id": "file-abc123"}]}]
        )
        result = cr.convert()
        assert "[image file: file-abc123]" in result

    def test_input_file_with_file_id_only_annotated(self):
        cr = _make_converter(
            [{"content": [{"type": "input_file", "file_id": "fid-xyz", "filename": "report.pdf"}]}]
        )
        result = cr.convert()
        assert "[file: report.pdf]" in result

    def test_input_file_with_file_data_not_annotated(self):
        """Files with data are materialised as attachments — no annotation."""
        cr = _make_converter(
            [{"content": [{"type": "input_file", "filename": "doc.txt", "file_data": _b64("hello")}]}]
        )
        result = cr.convert()
        assert "[file:" not in result

    def test_text_parts_still_extracted_alongside_non_text(self):
        cr = _make_converter(
            [
                {
                    "content": [
                        {"type": "input_text", "text": "Describe this image:"},
                        {"type": "input_image", "image_url": {"url": "https://img.example.com/x.png"}},
                    ]
                }
            ]
        )
        result = cr.convert()
        assert "Describe this image:" in result
        assert "[image: https://img.example.com/x.png]" in result


# ===========================================================================
# convert_attachments() — ConvertedAttachments
# ===========================================================================


@pytest.mark.unit
class TestConvertedAttachmentsNoContent:
    def test_no_attachments_for_string_input(self):
        ca = _make_converter("hello").convert_attachments()
        assert ca.attachments == []

    def test_no_attachments_for_none_input(self):
        ca = _make_converter(None).convert_attachments()
        assert ca.attachments == []

    def test_no_attachments_for_text_only_message(self):
        ca = _make_converter(
            [{"content": [{"type": "input_text", "text": "hi"}]}]
        ).convert_attachments()
        assert ca.attachments == []
        ca.cleanup()  # no-op, should not raise

    def test_no_attachments_for_external_image_url(self):
        """External URLs cannot be fetched — no attachment produced."""
        ca = _make_converter(
            [{"content": [{"type": "input_image", "image_url": {"url": "https://example.com/img.png"}}]}]
        ).convert_attachments()
        assert ca.attachments == []
        ca.cleanup()

    def test_no_attachments_for_file_id_only(self):
        """file_id without file_data cannot be materialised."""
        ca = _make_converter(
            [{"content": [{"type": "input_file", "file_id": "fid-123"}]}]
        ).convert_attachments()
        assert ca.attachments == []
        ca.cleanup()


@pytest.mark.unit
class TestConvertedAttachmentsFileData:
    def test_input_file_with_file_data_produces_attachment(self):
        ca = _make_converter(
            [{"content": [{"type": "input_file", "filename": "note.txt", "file_data": _b64("hello")}]}]
        ).convert_attachments()
        try:
            assert len(ca.attachments) == 1
            att = ca.attachments[0]
            assert att["type"] == "file"
            assert att["displayName"] == "note.txt"
            assert os.path.isfile(att["path"])
        finally:
            ca.cleanup()

    def test_file_contents_written_correctly(self):
        content = "important report data\nline two"
        ca = _make_converter(
            [{"content": [{"type": "input_file", "filename": "report.txt", "file_data": _b64(content)}]}]
        ).convert_attachments()
        try:
            path = ca.attachments[0]["path"]
            with open(path, "rb") as fh:
                assert fh.read() == content.encode()
        finally:
            ca.cleanup()

    def test_file_extension_inferred_from_filename(self):
        ca = _make_converter(
            [{"content": [{"type": "input_file", "filename": "data.csv", "file_data": _b64("a,b\n1,2")}]}]
        ).convert_attachments()
        try:
            assert ca.attachments[0]["path"].endswith(".csv")
        finally:
            ca.cleanup()

    def test_cleanup_deletes_temp_file(self):
        ca = _make_converter(
            [{"content": [{"type": "input_file", "filename": "x.txt", "file_data": _b64("x")}]}]
        ).convert_attachments()
        path = ca.attachments[0]["path"]
        assert os.path.isfile(path)
        ca.cleanup()
        assert not os.path.isfile(path)

    def test_cleanup_is_idempotent(self):
        ca = _make_converter(
            [{"content": [{"type": "input_file", "filename": "x.txt", "file_data": _b64("x")}]}]
        ).convert_attachments()
        ca.cleanup()
        ca.cleanup()  # second call must not raise

    def test_multiple_files_in_one_message(self):
        ca = _make_converter(
            [
                {
                    "content": [
                        {"type": "input_file", "filename": "a.txt", "file_data": _b64("aaa")},
                        {"type": "input_file", "filename": "b.txt", "file_data": _b64("bbb")},
                    ]
                }
            ]
        ).convert_attachments()
        try:
            assert len(ca.attachments) == 2
            assert {a["displayName"] for a in ca.attachments} == {"a.txt", "b.txt"}
        finally:
            ca.cleanup()

    def test_files_across_multiple_messages(self):
        ca = _make_converter(
            [
                {"content": [{"type": "input_file", "filename": "m1.txt", "file_data": _b64("1")}]},
                {"content": [{"type": "input_file", "filename": "m2.txt", "file_data": _b64("2")}]},
            ]
        ).convert_attachments()
        try:
            assert len(ca.attachments) == 2
        finally:
            ca.cleanup()


@pytest.mark.unit
class TestConvertedAttachmentsImages:
    def test_data_uri_png_produces_attachment(self):
        uri = _data_uri("image/png", "fake-png-data")
        ca = _make_converter(
            [{"content": [{"type": "input_image", "image_url": {"url": uri}}]}]
        ).convert_attachments()
        try:
            assert len(ca.attachments) == 1
            att = ca.attachments[0]
            assert att["type"] == "file"
            assert att["path"].endswith(".png")
            assert os.path.isfile(att["path"])
        finally:
            ca.cleanup()

    def test_data_uri_jpeg_produces_jpg_extension(self):
        uri = _data_uri("image/jpeg", "fake-jpeg-data")
        ca = _make_converter(
            [{"content": [{"type": "input_image", "image_url": {"url": uri}}]}]
        ).convert_attachments()
        try:
            assert ca.attachments[0]["path"].endswith(".jpg")
        finally:
            ca.cleanup()

    def test_image_contents_written_correctly(self):
        raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50  # fake PNG header
        uri = f"data:image/png;base64,{base64.b64encode(raw).decode()}"
        ca = _make_converter(
            [{"content": [{"type": "input_image", "image_url": {"url": uri}}]}]
        ).convert_attachments()
        try:
            with open(ca.attachments[0]["path"], "rb") as fh:
                assert fh.read() == raw
        finally:
            ca.cleanup()

    def test_image_url_as_plain_string(self):
        """image_url may be a plain string rather than a dict."""
        uri = _data_uri("image/gif", "fake-gif")
        ca = _make_converter(
            [{"content": [{"type": "input_image", "image_url": uri}]}]
        ).convert_attachments()
        try:
            assert len(ca.attachments) == 1
            assert ca.attachments[0]["path"].endswith(".gif")
        finally:
            ca.cleanup()

    def test_image_and_file_together(self):
        uri = _data_uri("image/png", "img-bytes")
        ca = _make_converter(
            [
                {
                    "content": [
                        {"type": "input_image", "image_url": {"url": uri}},
                        {"type": "input_file", "filename": "doc.pdf", "file_data": _b64("pdf-bytes")},
                    ]
                }
            ]
        ).convert_attachments()
        try:
            assert len(ca.attachments) == 2
            types_seen = {a["path"].split(".")[-1] for a in ca.attachments}
            assert "png" in types_seen
            assert "pdf" in types_seen
        finally:
            ca.cleanup()


@pytest.mark.unit
class TestConvertedAttachmentsBool:
    def test_bool_false_when_empty(self):
        ca = ConvertedAttachments(attachments=[])
        assert not ca

    def test_bool_true_when_has_attachments(self):
        ca = ConvertedAttachments(attachments=[{"type": "file", "path": "/tmp/x", "displayName": "x"}])
        assert ca


@pytest.mark.unit
class TestConvertedAttachmentsEdgeCases:
    """Edge-case tests for attachment materialisation."""

    def test_bad_base64_file_data_produces_no_attachment(self):
        """Invalid base64 in file_data should be silently skipped."""
        ca = _make_converter(
            [{"content": [{"type": "input_file", "filename": "bad.txt", "file_data": "!!!NOT-BASE64!!!"}]}]
        ).convert_attachments()
        assert ca.attachments == []
        ca.cleanup()

    def test_bad_base64_data_uri_produces_no_attachment(self):
        """Invalid base64 in image data URI should be silently skipped."""
        ca = _make_converter(
            [{"content": [{"type": "input_image", "image_url": {"url": "data:image/png;base64,!!!BAD!!!"}}]}]
        ).convert_attachments()
        assert ca.attachments == []
        ca.cleanup()

    def test_image_with_no_image_url_produces_no_attachment(self):
        """input_image part with missing image_url field should be skipped."""
        ca = _make_converter(
            [{"content": [{"type": "input_image"}]}]
        ).convert_attachments()
        assert ca.attachments == []

    def test_file_without_filename_uses_default(self):
        """input_file with file_data but no filename should use 'attachment' as default."""
        ca = _make_converter(
            [{"content": [{"type": "input_file", "file_data": _b64("hello")}]}]
        ).convert_attachments()
        try:
            assert len(ca.attachments) == 1
            assert ca.attachments[0]["displayName"] == "attachment"
        finally:
            ca.cleanup()

    def test_dict_input_with_content_key(self):
        """A single dict input with a 'content' string should work."""
        cr = _make_converter({"content": "Direct content"})
        result = cr.convert()
        assert result == "Direct content"

    def test_content_part_without_type(self):
        """A content part without 'type' should extract text if present."""
        cr = _make_converter(
            [{"content": [{"text": "fallback text"}]}]
        )
        result = cr.convert()
        assert "fallback text" in result

    def test_string_parts_in_content_list(self):
        """String items in a content list should be extracted as text."""
        cr = _make_converter(
            [{"content": ["raw string part", "another"]}]
        )
        result = cr.convert()
        assert "raw string part" in result
        assert "another" in result

    def test_non_dict_non_string_part_skipped(self):
        """Non-dict, non-string content parts should be skipped."""
        cr = _make_converter(
            [{"content": [42, True, {"type": "input_text", "text": "ok"}]}]
        )
        result = cr.convert()
        assert "ok" in result

    def test_image_url_as_plain_string(self):
        """image_url can be a plain string URL instead of a dict."""
        cr = _make_converter(
            [{"content": [{"type": "input_image", "image_url": "https://example.com/img.png"}]}]
        )
        result = cr.convert()
        assert "[image: https://example.com/img.png]" in result

    def test_content_not_list_or_string_returns_str(self):
        """Non-list, non-string content falls through to str()."""
        cr = _make_converter([{"content": 42}])
        result = cr.convert()
        assert "42" in result

    def test_input_file_with_file_id_only_annotates_text(self):
        """input_file with file_id but no file_data annotates prompt text."""
        cr = _make_converter(
            [{"content": [{"type": "input_file", "file_id": "file-123", "filename": "data.csv"}]}]
        )
        result = cr.convert()
        assert "[file: data.csv]" in result

    def test_content_list_in_attachments_skips_non_list_content(self):
        """Non-list content field on a message should be skipped in convert_attachments."""
        ca = _make_converter(
            [{"content": "just a string"}]
        ).convert_attachments()
        assert ca.attachments == []

    def test_cleanup_handles_already_deleted_file(self):
        """Cleanup should not raise if temp file already deleted."""
        ca = ConvertedAttachments(attachments=[], _temp_paths=["/nonexistent/path/12345"])
        ca.cleanup()  # should not raise
        assert ca._temp_paths == []

    def test_image_with_file_id_annotates_text(self):
        """input_image with file_id and no url annotates text."""
        cr = _make_converter(
            [{"content": [{"type": "input_image", "image_url": {"url": ""}, "file_id": "img-99"}]}]
        )
        result = cr.convert()
        assert "[image file: img-99]" in result
