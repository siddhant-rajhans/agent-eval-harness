import json

from agent.tools import (
    check_inventory,
    escalate_to_human,
    lookup_order,
    process_refund,
    update_shipping_address,
)


class TestLookupOrder:
    def test_valid_order(self):
        result = json.loads(lookup_order.invoke({"order_id": "ORD-1001"}))
        assert result["order_id"] == "ORD-1001"
        assert result["product_name"] == "Wireless Headphones"
        assert result["status"] == "delivered"

    def test_invalid_order(self):
        result = lookup_order.invoke({"order_id": "ORD-9999"})
        assert "not found" in result.lower()

    def test_all_orders_exist(self):
        for oid in ["ORD-1001", "ORD-1002", "ORD-1003", "ORD-1004", "ORD-1005"]:
            result = json.loads(lookup_order.invoke({"order_id": oid}))
            assert result["order_id"] == oid


class TestProcessRefund:
    def test_refund_delivered_order(self):
        result = json.loads(process_refund.invoke({"order_id": "ORD-1001", "reason": "Defective"}))
        assert result["status"] == "approved"
        assert result["refund_amount"] == 79.99

    def test_refund_cancelled_order(self):
        result = json.loads(process_refund.invoke({"order_id": "ORD-1004", "reason": "Changed mind"}))
        assert result["status"] == "denied"

    def test_refund_pending_order(self):
        result = json.loads(process_refund.invoke({"order_id": "ORD-1003", "reason": "Changed mind"}))
        assert result["status"] == "denied"

    def test_partial_refund(self):
        result = json.loads(process_refund.invoke({"order_id": "ORD-1002", "reason": "Partial return", "amount": 24.99}))
        assert result["status"] == "approved"
        assert result["refund_amount"] == 24.99

    def test_excess_refund(self):
        result = json.loads(process_refund.invoke({"order_id": "ORD-1005", "reason": "Defective", "amount": 200.0}))
        assert result["status"] == "denied"

    def test_invalid_order_refund(self):
        result = process_refund.invoke({"order_id": "ORD-9999", "reason": "Test"})
        assert "not found" in result.lower()


class TestEscalateToHuman:
    def test_escalation_creates_ticket(self):
        result = json.loads(escalate_to_human.invoke({"reason": "Customer upset", "context": "Multiple failed attempts"}))
        assert result["ticket_id"].startswith("ESC-")
        assert result["assigned_to"] == "Senior Support Agent"

    def test_escalation_with_priority(self):
        result = json.loads(escalate_to_human.invoke({"reason": "Security", "context": "Account breach", "priority": "high"}))
        assert "high" in result["message"]


class TestCheckInventory:
    def test_in_stock_product(self):
        result = json.loads(check_inventory.invoke({"product_id": "PROD-001"}))
        assert result["in_stock"] is True
        assert result["quantity_available"] > 0

    def test_out_of_stock_product(self):
        result = json.loads(check_inventory.invoke({"product_id": "PROD-003"}))
        assert result["in_stock"] is False
        assert result["estimated_restock"] is not None

    def test_invalid_product(self):
        result = check_inventory.invoke({"product_id": "PROD-999"})
        assert "not found" in result.lower()


class TestUpdateShippingAddress:
    def test_update_pending_order(self):
        result = update_shipping_address.invoke({"order_id": "ORD-1003", "new_address": "123 New St, Boston, MA"})
        assert "updated" in result.lower()

    def test_update_shipped_order(self):
        result = update_shipping_address.invoke({"order_id": "ORD-1002", "new_address": "456 New St"})
        assert "cannot" in result.lower() or "error" in result.lower()

    def test_update_delivered_order(self):
        result = update_shipping_address.invoke({"order_id": "ORD-1001", "new_address": "789 New St"})
        assert "cannot" in result.lower() or "error" in result.lower()
