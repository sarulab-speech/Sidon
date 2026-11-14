"""A time limit per job allocation for runs that resume across scheduler jobs (train.reset_time_on_resume)."""

from lightning.pytorch.callbacks import Timer


class AllocationTimer(Timer):
    @property
    def state_key(self) -> str:
        # Matches checkpoints written by Lightning's built-in max_time timer.
        return Timer.__qualname__

    def load_state_dict(self, state_dict: dict) -> None:
        # A new allocation has a new time budget; every other training state is restored as usual.
        super().load_state_dict({})
