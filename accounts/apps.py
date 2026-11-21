import sys
import threading
from django.apps import AppConfig


class AccountConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "accounts"

    def ready(self):
        # Do not run warmup during migrations or management commands
        cmd_args = sys.argv[1:] if len(sys.argv) > 1 else []
        skip_cmds = {"migrate", "makemigrations", "collectstatic", "check", "test", "showmigrations"}
        if any(cmd in skip_cmds for cmd in cmd_args):
            return

        def _warmup_worker():
            try:
                from .services.face_recognition import FaceRecognitionService
                service = FaceRecognitionService.get_instance()
                import numpy as np
                dummy = np.zeros((320, 320, 3), dtype=np.uint8)
                service.app.get(dummy)
            except Exception:
                pass

        warmup_thread = threading.Thread(target=_warmup_worker, daemon=True, name="InsightFaceWarmupThread")
        warmup_thread.start()
