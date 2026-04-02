"""Static lookup values for payment dataset fields.

These constants define the fixed vocabularies used when generating categorical
attributes such as payment status and account type.
"""

PAYMENT_CREDITOR_PREFIX = "CREDITOR"
"""Column-name prefix for creditor-side attributes."""

PAYMENT_DEBTOR_PREFIX = "DEBITOR"
"""Column-name prefix for debitor-side attributes."""

PAYMENT_STATUS = ("PENDING", "PROCESSING", "COMPLETED", "FAILED", "CANCELLED")
"""Valid payment status values."""

ACCOUNT_TYPES = ("SAVINGS", "CHECKING", "BUSINESS")
"""Valid account type values."""
