"""
Common API response schemas.

Provides a consistent response format for all API endpoints.
"""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorDetails(BaseModel):
    """Error details for failed API responses."""

    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional error context",
    )


class ApiResponse(BaseModel, Generic[T]):
    """
    Standard API response wrapper.

    All API endpoints return this format for consistency.

    Success response:
        {
            "success": true,
            "data": <response data>
        }

    Error response:
        {
            "success": false,
            "error": {
                "code": "ERROR_CODE",
                "message": "Human readable message",
                "details": {}
            }
        }
    """

    success: bool = Field(..., description="Whether the request succeeded")
    data: T | None = Field(default=None, description="Response data (on success)")
    error: ErrorDetails | None = Field(
        default=None,
        description="Error details (on failure)",
    )

    @classmethod
    def ok(cls, data: T) -> "ApiResponse[T]":
        """
        Create a successful response.

        Args:
            data: The response data.

        Returns:
            ApiResponse with success=True and the provided data.
        """
        return cls(success=True, data=data, error=None)

    @classmethod
    def fail(
        cls,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> "ApiResponse[None]":
        """
        Create a failed response.

        Args:
            code: Machine-readable error code.
            message: Human-readable error message.
            details: Additional error context.

        Returns:
            ApiResponse with success=False and error details.
        """
        return cls(
            success=False,
            data=None,
            error=ErrorDetails(
                code=code,
                message=message,
                details=details or {},
            ),
        )

    model_config = {"json_schema_extra": {"examples": [{"success": True, "data": {}}]}}


class PageResponse(ApiResponse):
    """`ApiResponse` plus `meta` — the paginated lists only (039 FR-39.15, FTID I-4).

    `meta` is `{"next": cursor | None, "limit": n}`. A SUBCLASS, not a field on `ApiResponse`:
    the shared envelope's generated schema is vendored by other repos (the weights API contract),
    and every unpaginated route keeps its exact bytes. Each item's shape is unchanged.
    """

    meta: dict[str, Any] | None = Field(default=None, description="Pagination: next cursor, limit")

    @classmethod
    def page(cls, data: Any, meta: dict[str, Any]) -> "PageResponse":
        return cls(success=True, data=data, error=None, meta=meta)


# ── 044: declarations for the published answer contract (`answers_contract.py`) ─────────────────

#: The marker `answers_contract` consumes (TD4): the field is not `required` in the contract.
MAY_BE_ABSENT = "x-may-be-absent"
#: The marker `answers_contract` consumes (TD14, IC-3): the stored value is checked in history form.
HISTORY_FORM = "x-history-form"


def may_be_absent(**kwargs: Any) -> Any:
    """A field some branch of its route LEAVES OUT (not one sent as null). Cite the branch beside each
    use. Everything else a model declares is `required` in the contract, because it is always sent
    (SH7, TD4)."""
    extra = {**kwargs.pop("json_schema_extra", {}), MAY_BE_ABSENT: True}
    return Field(default=None, json_schema_extra=extra, **kwargs)


def history_form(**kwargs: Any) -> Any:
    """A stored value that may predate its model (an approval's `result`, TD14): the contract keeps
    its types and drops every `required` below it."""
    extra = {**kwargs.pop("json_schema_extra", {}), HISTORY_FORM: True}
    return Field(default=None, json_schema_extra=extra, **kwargs)


class RefusalDetails(ErrorDetails):
    """A management refusal's `error` as sent (SH10): `ErrorDetails` plus the MCP tool that resolves it.

    Used only by the contract's `errors.management`, never by a route declaration, so the served
    `ErrorDetails` component (embedded in the frozen weights contract) keeps its bytes."""

    # Added by `exception_handlers.py` only when the exception names a `next_tool`.
    next: str | None = may_be_absent(description="The MCP tool that resolves this refusal")


class RefusalEnvelope(BaseModel):
    """The management envelope of a refusal: `success` false, `data` null, `error` set."""

    success: bool = Field(..., description="False for a refusal")
    data: None = Field(default=None, description="Always null on a refusal")
    error: RefusalDetails


class PageOf(PageResponse, Generic[T]):
    """`PageResponse` with typed items (044): `data` is the page's list. `PageResponse` itself is not
    changed, so its served component keeps its bytes."""

    data: list[T] | None = Field(default=None, description="The page's items")
