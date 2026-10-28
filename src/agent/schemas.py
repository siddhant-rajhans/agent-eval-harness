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


class RefundResult(BaseModel):
    order_id: str
    status: RefundStatus
    refund_amount: float
    message: str


class EscalationResult(BaseModel):
    ticket_id: str
    assigned_to: str
    estimated_wait: str
    message: str


class InventoryResult(BaseModel):
    product_id: str
    product_name: str
    in_stock: bool
    quantity_available: int
    estimated_restock: str | None = None


class AgentState(BaseModel):
    """Serializable agent state for evaluation results and persistence."""
    messages: list = Field(default_factory=list)
    tool_calls_made: list[str] = Field(default_factory=list)
    tasks_completed: list[str] = Field(default_factory=list)
    escalated: bool = False
    turn_count: int = 0
    total_tokens: int = 0
    retry_count: int = 0
    max_retries: int = 3
