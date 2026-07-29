import multiprocessing as mp
from auto_captioning.model_worker import run_worker


def _config_signature(caption_settings):
    """Keys that require a fresh process when they change."""
    # TODO: The signature will be different for an llama-cpp-python model and transformers model
    return (
        caption_settings.get('model_id'),
        str(caption_settings.get('device')),
        caption_settings.get('load_in_4_bit'),
        caption_settings.get('gpu_index'),
    )


class ModelSubprocessManager:
    def __init__(self):
        self._mp = mp.get_context('spawn')   # mandatory for CUDA
        self.process = None
        self.request_q = None
        self.response_q = None
        self.cancel_event = None
        self.current_signature = None

    def is_alive(self):
        return self.process is not None and self.process.is_alive()

    def ensure_model(self, caption_settings, models_directory_path,
                     tag_separator, log, on_loaded, on_load_error):
        signature = _config_signature(caption_settings)
        if self.is_alive() and signature == self.current_signature:
            self.cancel_event.clear()
            return True  # reuse the already-loaded model

        self.shutdown()  # kill old process -> frees all its VRAM
        self.request_q = self._mp.Queue()
        self.response_q = self._mp.Queue()
        self.cancel_event = self._mp.Event()
        self.process = self._mp.Process(
            target=run_worker,
            args=(self.request_q, self.response_q, caption_settings,
                  models_directory_path, tag_separator, self.cancel_event),
            daemon=True)
        self.process.start()
        self.current_signature = signature

        while True:
            kind, *rest = self.response_q.get()
            if kind == 'log':
                log(rest[0])
            elif kind == 'loaded':
                on_loaded()
                return True
            elif kind in ('load_error', 'fatal'):
                on_load_error(rest[0])
                self.shutdown()
                return False

    def caption(self, payload, log):
        self.request_q.put(('caption', payload))
        while True:
            kind, *rest = self.response_q.get()
            if kind == 'log':
                log(rest[0])
            elif kind == 'caption':
                return rest[0], rest[1]     # caption, console_output
            elif kind == 'caption_error':
                raise RuntimeError(rest[0])
            elif kind in ('fatal', 'exiting'):
                raise RuntimeError(rest[0] if rest else 'Worker exited')

    def cancel(self):
        if self.cancel_event is not None:
            self.cancel_event.set()

    def shutdown(self):
        if self.process is None:
            return
        try:
            if self.process.is_alive():
                self.request_q.put(('shutdown',))
                self.process.join(timeout=5)
        except Exception:
            pass
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        self.process = None
        self.current_signature = None
