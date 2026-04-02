from data_generation.static_data import field_static_data


class TestFieldStaticDataConstants:
    def test_payment_prefixes_are_uppercase_strings(self) -> None:
        assert isinstance(field_static_data.PAYMENT_CREDITOR_PREFIX, str)
        assert isinstance(field_static_data.PAYMENT_DEBTOR_PREFIX, str)
        assert field_static_data.PAYMENT_CREDITOR_PREFIX == field_static_data.PAYMENT_CREDITOR_PREFIX.upper()
        assert field_static_data.PAYMENT_DEBTOR_PREFIX == field_static_data.PAYMENT_DEBTOR_PREFIX.upper()

    def test_payment_prefixes_are_distinct(self) -> None:
        assert field_static_data.PAYMENT_CREDITOR_PREFIX != field_static_data.PAYMENT_DEBTOR_PREFIX

    def test_payment_status_is_non_empty_tuple(self) -> None:
        assert isinstance(field_static_data.PAYMENT_STATUS, tuple)
        assert len(field_static_data.PAYMENT_STATUS) > 0

    def test_payment_status_contains_expected_values(self) -> None:
        expected: set[str] = {
            "PENDING",
            "PROCESSING",
            "COMPLETED",
            "FAILED",
            "CANCELLED",
        }
        assert set(field_static_data.PAYMENT_STATUS) == expected

    def test_account_types_is_non_empty_tuple(self) -> None:
        assert isinstance(field_static_data.ACCOUNT_TYPES, tuple)
        assert len(field_static_data.ACCOUNT_TYPES) > 0

    def test_account_types_contains_expected_values(self) -> None:
        expected: set[str] = {"SAVINGS", "CHECKING", "BUSINESS"}
        assert set(field_static_data.ACCOUNT_TYPES) == expected
