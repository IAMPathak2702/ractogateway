from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError, conint

from ractogateway._models.chat import ChatConfig
from ractogateway._validation import (
    _build_correction_message,
    _format_validation_errors,
    validate_and_retry,
    with_inferred_response_model,
)
from ractogateway.adapters.base import LLMResponse
from ractogateway.prompts.engine import RactoPrompt


class MovieReview(BaseModel):
    title: str
    rating: conint(ge=1, le=10)
    pros: list[str]
    cons: list[str]
    recommend: bool


def _missing_title_payload() -> dict[str, Any]:
    return {
        "rating": 9,
        "pros": ["Strong performances", "Inventive visuals"],
        "cons": ["Complex exposition", "Uneven pacing"],
        "recommend": True,
    }


def _valid_payload() -> dict[str, Any]:
    return {
        "title": "Inception",
        "rating": 9,
        "pros": ["Strong performances", "Inventive visuals"],
        "cons": ["Complex exposition", "Uneven pacing"],
        "recommend": True,
    }


def test_format_validation_errors_missing_field_does_not_show_parent_dict() -> None:
    with pytest.raises(ValidationError) as exc_info:
        MovieReview.model_validate(_missing_title_payload())

    error_text = _format_validation_errors(exc_info.value)

    assert "title: Field required" in error_text
    assert (
        "This required field is missing from your response; add it with a valid value."
        in error_text
    )
    assert "your value:" not in error_text
    assert "{'rating': 9" not in error_text


def test_format_validation_errors_non_missing_shows_offending_value() -> None:
    with pytest.raises(ValidationError) as exc_info:
        MovieReview.model_validate(
            {
                "title": "Inception",
                "rating": 11,
                "pros": ["Strong performances", "Inventive visuals"],
                "cons": ["Complex exposition", "Uneven pacing"],
                "recommend": True,
            }
        )

    error_text = _format_validation_errors(exc_info.value)

    assert "rating" in error_text
    assert "your value: 11" in error_text


def test_build_correction_message_mentions_adding_missing_required_fields() -> None:
    message = _build_correction_message(
        bad_json='{"rating": 9}',
        error_text="  - title: Field required",
        attempt=1,
        model_name="MovieReview",
        required_fields=["title", "rating", "pros", "cons", "recommend"],
        missing_fields=["title"],
    )

    assert "Correct ONLY the fields listed below." in message
    assert "If a required field is missing, add it." in message
    assert "Fix ONLY the invalid fields" not in message
    assert "Required fields for MovieReview: title, rating, pros, cons, recommend." in message
    assert "Missing required fields to add now: title." in message


def test_validate_and_retry_uses_missing_field_actionable_prompt() -> None:
    config = ChatConfig(
        user_message="Review the movie Inception.",
        response_model=MovieReview,
        max_validation_retries=1,
    )
    seen_corrections: list[str] = []

    def adapter_run(correction_user_message: str) -> LLMResponse:
        seen_corrections.append(correction_user_message)
        valid = _valid_payload()
        return LLMResponse(content=json.dumps(valid), parsed=valid)

    invalid = _missing_title_payload()
    response = LLMResponse(content=json.dumps(invalid), parsed=invalid)

    validated = validate_and_retry(response, config, adapter_run=adapter_run)

    assert len(seen_corrections) == 1
    correction = seen_corrections[0]
    assert "title: Field required" in correction
    assert "If a required field is missing, add it." in correction
    assert "Required fields for MovieReview: title, rating, pros, cons, recommend." in correction
    assert "Missing required fields to add now: title." in correction
    assert "your value:" not in correction
    assert "{'rating': 9" not in correction
    assert isinstance(validated.parsed, dict)
    assert validated.parsed["title"] == "Inception"


def test_validate_and_retry_recovers_when_parsed_is_none_but_content_has_json() -> None:
    config = ChatConfig(
        user_message="Review the movie Inception.",
        response_model=MovieReview,
        max_validation_retries=1,
    )
    seen_corrections: list[str] = []

    def adapter_run(correction_user_message: str) -> LLMResponse:
        seen_corrections.append(correction_user_message)
        valid = _valid_payload()
        # Simulate adapters returning content without parsed dict populated.
        return LLMResponse(content=json.dumps(valid), parsed=None)

    invalid = _missing_title_payload()
    response = LLMResponse(content=json.dumps(invalid), parsed=None)

    validated = validate_and_retry(response, config, adapter_run=adapter_run)

    assert len(seen_corrections) == 1
    assert isinstance(validated.parsed, dict)
    assert validated.parsed["title"] == "Inception"


def test_validate_and_retry_autofills_missing_title_after_retries_exhausted() -> None:
    config = ChatConfig(
        user_message="Review the movie Inception.",
        response_model=MovieReview,
        max_validation_retries=0,
    )
    invalid = _missing_title_payload()
    response = LLMResponse(content=json.dumps(invalid), parsed=invalid)

    def adapter_run(_: str) -> LLMResponse:
        pytest.fail("adapter_run should not be called when max_validation_retries=0")

    validated = validate_and_retry(response, config, adapter_run=adapter_run)

    assert isinstance(validated.parsed, dict)
    assert validated.parsed["title"] == "Inception"
    assert validated.parsed["rating"] == 9


def test_validate_and_retry_autofills_missing_constrained_int_from_bounds() -> None:
    config = ChatConfig(
        user_message="Review the movie Inception.",
        response_model=MovieReview,
        max_validation_retries=0,
    )
    invalid = {
        "title": "Inception",
        "pros": ["Strong performances", "Inventive visuals"],
        "cons": ["Complex exposition", "Uneven pacing"],
        "recommend": True,
    }
    response = LLMResponse(content=json.dumps(invalid), parsed=invalid)

    def adapter_run(_: str) -> LLMResponse:
        pytest.fail("adapter_run should not be called when max_validation_retries=0")

    validated = validate_and_retry(response, config, adapter_run=adapter_run)

    assert isinstance(validated.parsed, dict)
    assert validated.parsed["rating"] == 1


def test_validate_and_retry_autofills_after_all_retries_exhausted() -> None:
    config = ChatConfig(
        user_message="Review the movie Inception.",
        response_model=MovieReview,
        max_validation_retries=2,
    )
    invalid = _missing_title_payload()
    response = LLMResponse(content=json.dumps(invalid), parsed=invalid)
    call_count = 0

    def adapter_run(_: str) -> LLMResponse:
        nonlocal call_count
        call_count += 1
        return LLMResponse(content=json.dumps(invalid), parsed=invalid)

    validated = validate_and_retry(response, config, adapter_run=adapter_run)

    assert call_count == 2
    assert isinstance(validated.parsed, dict)
    assert validated.parsed["title"] == "Inception"


def test_with_inferred_response_model_uses_prompt_output_model_when_unset() -> None:
    config = ChatConfig(user_message="Review Titanic.")
    prompt = RactoPrompt(
        role="Film critic",
        aim="Write a review",
        constraints=["Return JSON only"],
        tone="Critical",
        output_format=MovieReview,
    )

    resolved = with_inferred_response_model(config, prompt)

    assert resolved is not config
    assert resolved.response_model is MovieReview
    assert config.response_model is None


def test_with_inferred_response_model_keeps_explicit_response_model() -> None:
    class OtherModel(BaseModel):
        value: int

    config = ChatConfig(user_message="x", response_model=OtherModel)
    prompt = RactoPrompt(
        role="r",
        aim="a",
        constraints=["c"],
        tone="t",
        output_format=MovieReview,
    )

    resolved = with_inferred_response_model(config, prompt)

    assert resolved is config
    assert resolved.response_model is OtherModel
