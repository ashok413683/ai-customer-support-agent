from langchain_core.tools import tool
import random


@tool
def create_support_ticket(
    customer_name: str,
    customer_email: str,
    issue: str
) -> str:
    """
    Create a support ticket when the customer needs human support.
    """

    ticket_id = f"TICKET-{random.randint(1000, 9999)}"

    return (
        f"Support ticket created successfully. "
        f"Ticket ID: {ticket_id}. "
        f"Customer: {customer_name}. "
        f"Our support team will contact you shortly."
    )