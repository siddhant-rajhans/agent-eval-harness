from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from agent.schemas import (
    EscalationResult,
    InventoryResult,
    Order,
    OrderStatus,
    RefundResult,
    RefundStatus,
)

_FIXTURES_PATH = Path(__file__).parent / "fixtures.json"

def _load_fixtures() -> dict[str, Any]:
    with open(_FIXTURES_PATH) as f:
        return json.load(f)

_DATA = _load_fixtures()
MOCK_ORDERS: dict[str, dict[str, Any]] = _DATA["orders"]
MOCK_INVENTORY: dict[str, dict[str, Any]] = _DATA["inventory"]


@tool
def lookup_order(order_id: str) -> str:
    """Look up order details by order ID. Returns order status, items, and shipping info."""
    if order_id not in MOCK_ORDERS:
        return f"Error: Order {order_id} not found. Please verify the order ID."
    order = Order(**MOCK_ORDERS[order_id])
    return order.model_dump_json(indent=2)


@tool
def process_refund(order_id: str, reason: str, amount: float | None = None) -> str:
    """Process a refund for an order. Provide the order ID and reason. Optionally specify a partial refund amount."""
    if order_id not in MOCK_ORDERS:
        return f"Error: Order {order_id} not found."
    order_data = MOCK_ORDERS[order_id]
    status = OrderStatus(order_data["status"])

    if status == OrderStatus.CANCELLED:
        return RefundResult(
            order_id=order_id,
            status=RefundStatus.DENIED,
            refund_amount=0,
            message="Order was already cancelled. No refund applicable.",
        ).model_dump_json()

    if status == OrderStatus.PENDING:
        return RefundResult(
            order_id=order_id,
            status=RefundStatus.DENIED,
            refund_amount=0,
            message="Order has not shipped yet. Please cancel the order instead.",
        ).model_dump_json()

    refund_amount = amount if amount is not None else order_data["total_amount"]
    if refund_amount > order_data["total_amount"]:
        return RefundResult(
            order_id=order_id,
            status=RefundStatus.DENIED,
            refund_amount=0,
            message=f"Refund amount ${refund_amount:.2f} exceeds order total ${order_data['total_amount']:.2f}.",
        ).model_dump_json()

    return RefundResult(
        order_id=order_id,
        status=RefundStatus.APPROVED,
        refund_amount=refund_amount,
        message=f"Refund of ${refund_amount:.2f} approved for order {order_id}. Reason: {reason}. "
        f"Expected processing time: 3-5 business days.",
    ).model_dump_json()


@tool
def escalate_to_human(reason: str, context: str, priority: str = "medium") -> str:
    """Escalate the conversation to a human agent. Use when you cannot resolve the issue or the customer requests it."""
    ticket_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"
    return EscalationResult(
        ticket_id=ticket_id,
        assigned_to="Senior Support Agent",
        estimated_wait="5-10 minutes",
        message=f"Ticket {ticket_id} created with {priority} priority. "
        f"A human agent will review: {reason}",
    ).model_dump_json()


@tool
def check_inventory(product_id: str) -> str:
    """Check product availability and stock levels."""
    if product_id not in MOCK_INVENTORY:
        return f"Error: Product {product_id} not found in catalog."
    result = InventoryResult(**MOCK_INVENTORY[product_id])
    return result.model_dump_json(indent=2)


@tool
def update_shipping_address(order_id: str, new_address: str) -> str:
    """Update the shipping address for an order. Only works for orders that haven't shipped yet."""
    if order_id not in MOCK_ORDERS:
        return f"Error: Order {order_id} not found."
    order_data = MOCK_ORDERS[order_id]
    status = OrderStatus(order_data["status"])

    if status in (OrderStatus.SHIPPED, OrderStatus.DELIVERED):
        return f"Error: Cannot update address for order {order_id}. Status: {status.value}. Package already in transit."
    if status == OrderStatus.CANCELLED:
        return f"Error: Order {order_id} has been cancelled."

    return (
        f"Shipping address updated for order {order_id}. "
        f"New address: {new_address}. Previous address: {order_data['shipping_address']}."
    )


ALL_TOOLS = [lookup_order, process_refund, escalate_to_human, check_inventory, update_shipping_address]
