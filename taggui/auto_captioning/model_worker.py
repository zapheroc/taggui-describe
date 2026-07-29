# auto_captioning/model_worker.py
import sys
import traceback
from pathlib import Path


class _QueueWriter:
    """Redirects stdout/stderr into the response queue as log messages."""
    def __init__(self, queue):
        self._queue = queue
    def write(self, text):
        if text:
            self._queue.put(('log', text))
    def flush(self):
        pass
    def isatty(self):
        return False


def run_worker(request_q, response_q, caption_settings,
               models_directory_path, tag_separator, cancel_event):
    # Redirect output FIRST so model-load logs are captured too.
    sys.stdout = _QueueWriter(response_q)
    sys.stderr = _QueueWriter(response_q)
    try:
        from auto_captioning.models_list import get_model_class
        from auto_captioning.worker_context import WorkerContext
        from utils.image import Image  # adapt import/ctor to your codebase ?????? TODO what does this mean...

        ctx = WorkerContext(models_directory_path, tag_separator, cancel_event)
        model_class = get_model_class(caption_settings['model_id'])
        model = model_class(captioning_thread_=ctx,
                            caption_settings=caption_settings)

        error_message = model.get_error_message()
        if error_message:
            response_q.put(('load_error', error_message))
            return

        model.load_processor_and_model()
        response_q.put(('loaded',))

        while True:
            request = request_q.get()
            if request is None or request[0] == 'shutdown':
                break
            if request[0] == 'caption':
                data = request[1]
                try:
                    image = Image(path=Path(data['path']))  # adapt ctor TODO what does this mean?
                    image.tags = data.get('tags', [])
                    image.description = data.get('description', '')
                    prompt = model.get_image_prompt(image)
                    model_inputs = model.get_model_inputs(prompt, image)
                    caption, console_output = model.generate_caption(
                        model_inputs, prompt)
                    del model_inputs
                    response_q.put(('caption', caption, console_output))
                except Exception as exception:
                    response_q.put(
                        ('caption_error', f'{data.get("path")}: {exception}'))
    except Exception:
        response_q.put(('fatal', traceback.format_exc()))
    finally:
        response_q.put(('exiting',))
