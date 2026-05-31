# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

from nvflare.fuel.utils.validation_utils import check_positive_number

DIRECTION_TASK_PAYLOAD_DOWNLOAD = "task_payload_download"
DIRECTION_RESULT_UPLOAD = "result_upload"
VALID_TRANSFER_DIRECTIONS = {DIRECTION_TASK_PAYLOAD_DOWNLOAD, DIRECTION_RESULT_UPLOAD}

STREAMING_IDLE_TIMEOUT = "streaming_idle_timeout"
STREAMING_MAX_PEER_SILENCE = "streaming_max_peer_silence"

DEFAULT_STREAMING_IDLE_TIMEOUT = 600.0
DEFAULT_STREAMING_MAX_PEER_SILENCE = 900.0
STREAMING_MAX_PEER_SILENCE_IDLE_MULTIPLIER = 1.5
STREAM_PROGRESS_COMPLETION_ACK_GRACE = 30.0
_RECEIVER_ID_UNSET = object()


class TransferProgressState:
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"

    TERMINAL_STATES = {COMPLETED, FAILED, ABORTED}
    VALID_STATES = {ACTIVE, COMPLETED, FAILED, ABORTED}


TransferProgressKey = Tuple[str, str, str, str, Optional[str]]


@dataclass(frozen=True)
class StreamingProgressConfig:
    streaming_idle_timeout: float = DEFAULT_STREAMING_IDLE_TIMEOUT
    streaming_max_peer_silence: float = DEFAULT_STREAMING_MAX_PEER_SILENCE


@dataclass
class TransferProgressRecord:
    job_id: str
    task_id: str
    transfer_id: str
    direction: str
    receiver_id: Optional[str]
    sequence: int
    bytes_done: int
    items_done: Optional[int]
    started_time: float
    last_progress_time: float
    state: str = TransferProgressState.ACTIVE
    transfer_id_kind: Optional[str] = None

    @property
    def key(self) -> TransferProgressKey:
        return self.job_id, self.task_id, self.transfer_id, self.direction, self.receiver_id

    @property
    def terminal(self) -> bool:
        return self.state in TransferProgressState.TERMINAL_STATES


@dataclass(frozen=True)
class TransferProgressUpdate:
    accepted: bool
    progressed: bool
    record: Optional[TransferProgressRecord]
    reason: str = ""


def resolve_streaming_progress_config(
    config: Optional[Mapping[str, object]] = None,
    *,
    streaming_idle_timeout: Optional[float] = None,
    streaming_max_peer_silence: Optional[float] = None,
) -> StreamingProgressConfig:
    """Resolve generic streaming progress timeouts.

    Explicit keyword values take precedence over mapping values.  When the idle
    timeout is raised above the default and max peer silence is not explicit,
    derive max silence from the idle timeout.
    """

    config = config or {}
    idle_explicit = streaming_idle_timeout is not None or STREAMING_IDLE_TIMEOUT in config
    if streaming_idle_timeout is None:
        streaming_idle_timeout = config.get(STREAMING_IDLE_TIMEOUT, DEFAULT_STREAMING_IDLE_TIMEOUT)
    if streaming_max_peer_silence is None:
        streaming_max_peer_silence = config.get(STREAMING_MAX_PEER_SILENCE)

    check_positive_number(STREAMING_IDLE_TIMEOUT, streaming_idle_timeout)
    streaming_idle_timeout = float(streaming_idle_timeout)

    if streaming_max_peer_silence is None:
        if idle_explicit and streaming_idle_timeout > DEFAULT_STREAMING_IDLE_TIMEOUT:
            streaming_max_peer_silence = max(
                DEFAULT_STREAMING_MAX_PEER_SILENCE,
                STREAMING_MAX_PEER_SILENCE_IDLE_MULTIPLIER * streaming_idle_timeout,
            )
        else:
            streaming_max_peer_silence = DEFAULT_STREAMING_MAX_PEER_SILENCE
    else:
        check_positive_number(STREAMING_MAX_PEER_SILENCE, streaming_max_peer_silence)

    return StreamingProgressConfig(
        streaming_idle_timeout=streaming_idle_timeout,
        streaming_max_peer_silence=float(streaming_max_peer_silence),
    )


class TransferProgressTracker:
    """Direction-neutral monotonic progress tracker for streamed transfers."""

    def __init__(
        self,
        *,
        idle_timeout: float = DEFAULT_STREAMING_IDLE_TIMEOUT,
        clock: Optional[Callable[[], float]] = None,
    ):
        check_positive_number("idle_timeout", idle_timeout)
        self.idle_timeout = float(idle_timeout)
        self._clock = clock or time.time
        self._records: Dict[TransferProgressKey, TransferProgressRecord] = {}

    def update(
        self,
        *,
        job_id: str,
        task_id: str,
        transfer_id: str,
        direction: str,
        sequence: int,
        bytes_done: int,
        items_done: Optional[int] = None,
        state: str = TransferProgressState.ACTIVE,
        transfer_id_kind: Optional[str] = None,
        receiver_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> TransferProgressUpdate:
        self._validate_direction(direction)
        self._validate_update(sequence=sequence, bytes_done=bytes_done, items_done=items_done, state=state)

        now = self._clock() if timestamp is None else float(timestamp)
        receiver_id = self._normalize_receiver_id(direction, receiver_id)
        key = self._make_key(
            job_id=job_id,
            task_id=task_id,
            transfer_id=transfer_id,
            direction=direction,
            receiver_id=receiver_id,
        )
        record = self._records.get(key)

        if record is None:
            new_record = TransferProgressRecord(
                job_id=job_id,
                task_id=task_id,
                transfer_id=transfer_id,
                direction=direction,
                receiver_id=receiver_id,
                sequence=sequence,
                bytes_done=bytes_done,
                items_done=items_done,
                started_time=now,
                last_progress_time=now,
                state=state,
                transfer_id_kind=transfer_id_kind,
            )
            self._records[key] = new_record
            return TransferProgressUpdate(accepted=True, progressed=True, record=new_record)

        if record.terminal:
            return TransferProgressUpdate(accepted=False, progressed=False, record=record, reason="terminal")

        if sequence <= record.sequence:
            return TransferProgressUpdate(accepted=False, progressed=False, record=record, reason="stale_sequence")

        if bytes_done < record.bytes_done:
            return TransferProgressUpdate(accepted=False, progressed=False, record=record, reason="bytes_regressed")

        if record.items_done is not None and items_done is not None and items_done < record.items_done:
            return TransferProgressUpdate(accepted=False, progressed=False, record=record, reason="items_regressed")

        next_items_done = self._next_items_done(record.items_done, items_done)
        counters_advanced = bytes_done > record.bytes_done or self._items_advanced(record.items_done, next_items_done)
        terminal_observed = state in TransferProgressState.TERMINAL_STATES
        progressed = counters_advanced or terminal_observed

        record.sequence = sequence
        record.bytes_done = max(record.bytes_done, bytes_done)
        record.items_done = next_items_done
        record.state = state
        if transfer_id_kind and not record.transfer_id_kind:
            record.transfer_id_kind = transfer_id_kind
        if progressed:
            record.last_progress_time = max(record.last_progress_time, now)

        return TransferProgressUpdate(accepted=True, progressed=progressed, record=record)

    def get_record(
        self,
        *,
        job_id: str,
        task_id: str,
        transfer_id: str,
        direction: str,
        receiver_id: Optional[str] = None,
    ) -> Optional[TransferProgressRecord]:
        return self._records.get(
            self._make_key(
                job_id=job_id,
                task_id=task_id,
                transfer_id=transfer_id,
                direction=direction,
                receiver_id=receiver_id,
            )
        )

    def records(
        self,
        *,
        job_id: Optional[str] = None,
        task_id: Optional[str] = None,
        direction: Optional[str] = None,
        receiver_id: object = _RECEIVER_ID_UNSET,
    ) -> Iterable[TransferProgressRecord]:
        filter_receiver = receiver_id is not _RECEIVER_ID_UNSET
        normalized_receiver_id = None
        if filter_receiver:
            normalized_receiver_id = self._normalize_receiver_id(direction, receiver_id) if direction else receiver_id
        return [
            record
            for record in self._records.values()
            if (job_id is None or record.job_id == job_id)
            and (task_id is None or record.task_id == task_id)
            and (direction is None or record.direction == direction)
            and (not filter_receiver or record.receiver_id == normalized_receiver_id)
        ]

    def is_stalled(
        self,
        *,
        job_id: str,
        task_id: str,
        transfer_id: str,
        direction: str,
        receiver_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> bool:
        record = self.get_record(
            job_id=job_id,
            task_id=task_id,
            transfer_id=transfer_id,
            direction=direction,
            receiver_id=receiver_id,
        )
        if record is None or record.terminal:
            return False
        return self._is_record_stalled(record, self._clock() if now is None else now)

    def stalled_records(
        self,
        now: Optional[float] = None,
        *,
        job_id: Optional[str] = None,
        task_id: Optional[str] = None,
        direction: Optional[str] = None,
        receiver_id: object = _RECEIVER_ID_UNSET,
    ) -> Iterable[TransferProgressRecord]:
        check_time = self._clock() if now is None else now
        return [
            record
            for record in self.records(job_id=job_id, task_id=task_id, direction=direction, receiver_id=receiver_id)
            if self._is_record_stalled(record, check_time)
        ]

    def mark_terminal(
        self,
        *,
        job_id: str,
        task_id: str,
        transfer_id: str,
        direction: str,
        state: str,
        sequence: Optional[int] = None,
        receiver_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> TransferProgressUpdate:
        if state not in TransferProgressState.TERMINAL_STATES:
            raise ValueError(f"terminal state must be one of {TransferProgressState.TERMINAL_STATES}, but got {state}")

        record = self.get_record(
            job_id=job_id,
            task_id=task_id,
            transfer_id=transfer_id,
            direction=direction,
            receiver_id=receiver_id,
        )
        if record is None:
            if sequence is None:
                sequence = 0
            return self.update(
                job_id=job_id,
                task_id=task_id,
                transfer_id=transfer_id,
                direction=direction,
                receiver_id=receiver_id,
                sequence=sequence,
                bytes_done=0,
                items_done=None,
                state=state,
                timestamp=timestamp,
            )

        if sequence is None:
            sequence = record.sequence + 1
        return self.update(
            job_id=job_id,
            task_id=task_id,
            transfer_id=transfer_id,
            direction=direction,
            receiver_id=receiver_id,
            sequence=sequence,
            bytes_done=record.bytes_done,
            items_done=record.items_done,
            state=state,
            transfer_id_kind=record.transfer_id_kind,
            timestamp=timestamp,
        )

    def prune(
        self,
        *,
        before_time: Optional[float] = None,
        include_active: bool = False,
    ) -> int:
        if before_time is None:
            before_time = self._clock()

        keys_to_remove = [
            key
            for key, record in self._records.items()
            if (include_active or record.terminal) and record.last_progress_time <= before_time
        ]
        for key in keys_to_remove:
            del self._records[key]
        return len(keys_to_remove)

    def remove(
        self,
        *,
        job_id: str,
        task_id: str,
        transfer_id: str,
        direction: str,
        receiver_id: Optional[str] = None,
    ) -> bool:
        return (
            self._records.pop(
                self._make_key(
                    job_id=job_id,
                    task_id=task_id,
                    transfer_id=transfer_id,
                    direction=direction,
                    receiver_id=receiver_id,
                ),
                None,
            )
            is not None
        )

    def clear(self):
        self._records.clear()

    @staticmethod
    def _validate_update(*, sequence: int, bytes_done: int, items_done: Optional[int], state: str):
        if not isinstance(sequence, int):
            raise TypeError(f"sequence must be an int, but got {type(sequence)}.")
        if sequence < 0:
            raise ValueError(f"sequence must >= 0, but got {sequence}")
        if not isinstance(bytes_done, int):
            raise TypeError(f"bytes_done must be an int, but got {type(bytes_done)}.")
        if bytes_done < 0:
            raise ValueError(f"bytes_done must >= 0, but got {bytes_done}")
        if items_done is not None:
            if not isinstance(items_done, int):
                raise TypeError(f"items_done must be an int, but got {type(items_done)}.")
            if items_done < 0:
                raise ValueError(f"items_done must >= 0, but got {items_done}")
        if state not in TransferProgressState.VALID_STATES:
            raise ValueError(f"state must be one of {TransferProgressState.VALID_STATES}, but got {state}")

    @staticmethod
    def _validate_direction(direction: str):
        if direction not in VALID_TRANSFER_DIRECTIONS:
            raise ValueError(f"direction must be one of {VALID_TRANSFER_DIRECTIONS}, but got {direction}")

    @staticmethod
    def _next_items_done(current_items_done: Optional[int], update_items_done: Optional[int]) -> Optional[int]:
        if current_items_done is None:
            return update_items_done
        if update_items_done is None:
            return current_items_done
        return max(current_items_done, update_items_done)

    @staticmethod
    def _items_advanced(current_items_done: Optional[int], next_items_done: Optional[int]) -> bool:
        if next_items_done is None:
            return False
        if current_items_done is None:
            return next_items_done > 0
        return next_items_done > current_items_done

    @staticmethod
    def _normalize_receiver_id(direction: Optional[str], receiver_id: Optional[str]) -> Optional[str]:
        if direction != DIRECTION_RESULT_UPLOAD:
            return None
        return None if receiver_id is None else str(receiver_id)

    @classmethod
    def _make_key(
        cls,
        *,
        job_id: str,
        task_id: str,
        transfer_id: str,
        direction: str,
        receiver_id: Optional[str] = None,
    ) -> TransferProgressKey:
        return job_id, task_id, transfer_id, direction, cls._normalize_receiver_id(direction, receiver_id)

    def _is_record_stalled(self, record: TransferProgressRecord, now: float) -> bool:
        return not record.terminal and now - record.last_progress_time >= self.idle_timeout
