from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class OrderStatus(str, Enum):
    PENDING = "pending"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    RETURNED = "returned"


class RefundStatus(str, Enum):
    APPROVED = "approved"
    DENIED = "denied"
    PENDING_REVIEW = "pending_review"


class Order(BaseModel):
    order_id: str
    customer_id: str
    product_name: str
    quantity: int
    total_amount: float
    status: OrderStatus
    shipping_address: str
    tracking_number: str | None = None
    order_date: str


class RefundRequest(BaseModel):
    order_id: str = Field(description="The order ID to refund")
    reason: str = Field(description="Reason for the refund request")
    amount: float | None = Field(default=None, description="Partial refund amount; None for full refund")


class RefundResult(BaseModel):
    order_id: str
    status: RefundStatus
    refund_amount: float
    message: str


class EscalationRequest(BaseModel):
    reason: str = Field(description="Why this needs human intervention")
    context: str = Field(description="Summary of the conversation so far")
    priority: str = Field(default="medium", description="Priority level: low, medium, high")


class EscalationResult(BaseModel):
    ticket_id: str
    assigned_to: str
    estimated_wait: str
    message: str


class ShippingUpdate(BaseModel):
    order_id: str = Field(description="The order to update")
    new_address: str = Field(description="New shipping address")


class InventoryCheck(BaseModel):
    product_id: str = Field(description="Product ID to check")


class InventoryResult(BaseModel):
    product_id: str
    product_name: str
    in_stock: bool
    quantity_available: int
    estimated_restock: str | None = None


class AgentState(BaseModel):
    """State tracked through the conversation graph."""
    messages: list = Field(default_factory=list)
    tool_calls_made: list[str] = Field(default_factory=list)
    tasks_completed: list[str] = Field(default_factory=list)
    escalated: bool = False
    turn_count: int = 0
    total_tokens: int = 0
    retry_count: int = 0
    max_retries: int = 3
