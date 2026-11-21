"""
Voice Analysis Views for VoiceStressAI / SVARAABHAVA.

Handles:
- Voice Recording Page (/voice/)
- Audio Analysis API Endpoint (/voice/analyze/)
- Emotion & Stress Analysis Dashboard (/voice/results/)
"""

import json
import logging
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.http import require_http_methods, require_POST
from django.views.decorators.csrf import ensure_csrf_cookie

from .services.voice_models import analyze_voice_recording, AudioProcessingError

logger = logging.getLogger(__name__)


@ensure_csrf_cookie
def voice_record_page(request):
    """
    Renders the dark midnight voice recording page (index.html).
    Allows live 10-second microphone recording or audio file upload.
    """
    context = {
        "active_tab": "record",
        "duration_seconds": 10.0,
    }
    return render(request, "audio.html", context)


@require_POST
def voice_analyze_api(request):
    """
    Receives multipart audio file from frontend, runs the 3 independent models
    (Stress, Strain, Emotion), saves results to user session, and returns JSON.
    """
    if "audio" not in request.FILES and "audio_file" not in request.FILES:
        return JsonResponse({
            "success": False,
            "error": "No audio file provided. Please record or upload an audio file.",
        }, status=400)

    audio_file = request.FILES.get("audio") or request.FILES.get("audio_file")

    # Limit file size to 25MB
    if audio_file.size > 25 * 1024 * 1024:
        return JsonResponse({
            "success": False,
            "error": "Audio file exceeds maximum size limit of 25MB.",
        }, status=400)

    try:
        file_bytes = audio_file.read()
        filename = audio_file.name or "voice_recording.wav"

        # Execute the unified 3-model analysis pipeline
        result = analyze_voice_recording(file_bytes, filename=filename)

        # Persist latest analysis in session
        request.session["latest_voice_analysis"] = result
        request.session.modified = True

        return JsonResponse({
            "success": True,
            "message": "Voice analysis completed successfully.",
            "redirect_url": "/voice-analysis/results/",
            "data": result,
        })

    except AudioProcessingError as e:
        logger.warning("Audio processing error during voice analysis: %s", e)
        return JsonResponse({
            "success": False,
            "error": str(e),
        }, status=400)

    except Exception as e:
        logger.exception("Unexpected error during voice analysis: %s", e)
        return JsonResponse({
            "success": False,
            "error": f"Voice analysis failed: {str(e)}",
        }, status=500)


def voice_results_page(request):
    """
    Renders the Emotion, Stress & Strain Dashboard (emotion_analysis.html).
    Retrieves the latest analysis from session. If no recording is present,
    presents a clean prompt directing the user to record first.
    """
    analysis = request.session.get("latest_voice_analysis")
    has_recording = bool(analysis and analysis.get("success"))

    context = {
        "active_tab": "results",
        "has_recording": has_recording,
        "analysis": analysis if has_recording else None,
        "analysis_json": json.dumps(analysis) if has_recording else "null",
    }

    return render(request, "emotion_analysis.html", context)
