from tools import create_support_ticket

result = create_support_ticket.invoke({
    "customer_name": "Ashok",
    "customer_email": "ashok@example.com",
    "issue": "My refund has not arrived."
})

print(result)