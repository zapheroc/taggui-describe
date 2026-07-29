# auto_captioning/worker_context.py
class _SignalStub:
    def emit(self, *args, **kwargs):
        pass

class WorkerContext:
    """Replaces CaptioningThread + AutoCaptioner inside the subprocess."""
    def __init__(self, models_directory_path, tag_separator, cancel_event):
        # formerly on the QThread
        self.models_directory_path = models_directory_path
        self.tag_separator = tag_separator
        self._cancel_event = cancel_event
        self.clear_console_text_edit_requested = _SignalStub()
        # formerly on AutoCaptioner (the per-process model cache)
        self.processor = None
        self.model = None
        self.model_id = None
        self.model_device_type = None
        self.is_model_loaded_in_4_bit = None

    @property
    def is_canceled(self):
        return self._cancel_event.is_set()

    def parent(self):
        return self
