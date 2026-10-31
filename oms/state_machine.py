"""Trade lifecycle. Every status change goes through transition(), so an impossible jump
(for example CLOSED back to OPEN) is an error instead of a silent corruption."""

TRANSITIONS = {
    "PENDING_APPROVAL": {"APPROVED", "REJECTED", "EXPIRED", "CANCELLED"},
    "APPROVED": {"ENTRY_PLACED", "REJECTED", "ERROR", "CANCELLED"},
    "ENTRY_PLACED": {"OPEN", "REJECTED", "CANCELLED", "ERROR"},
    "OPEN": {"EXITING", "CLOSED"},
    "EXITING": {"CLOSED", "OPEN", "ERROR"},
    "CLOSED": set(),
    "REJECTED": set(),
    "EXPIRED": set(),
    "CANCELLED": set(),
    "ERROR": {"CLOSED"},
}


class InvalidTransition(Exception):
    pass


def transition(trade, new_status: str, note: str = ""):
    old = trade.status
    if new_status not in TRANSITIONS.get(old, set()):
        raise InvalidTransition(f"{trade.symbol} {trade.id}: {old} -> {new_status} is not allowed")
    trade.status = new_status
    trade.log(f"{old} -> {new_status}" + (f": {note}" if note else ""))
