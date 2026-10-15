# auto_captioning/worker_context.py
class WorkerContext:
    """Replaces CaptioningThread + AutoCaptioner inside the subprocess."""
    def __init__(self, models_directory_path, tag_separator, cancel_event):
        # formerly on the QThread
        self.models_directory_path = models_directory_path
        self.tag_separator = tag_separator
        self._cancel_event = cancel_event

    @property
    def is_canceled(self):
        return self._cancel_event.is_set()
