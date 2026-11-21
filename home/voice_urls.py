"""
Voice Analysis URL routing for SVARAABHAVA / VoiceStressAI (Feature Sub-Routing).
"""

from django.urls import path
from . import voice_views

urlpatterns = [
    path("", voice_views.voice_record_page, name="voice_home"),
    path("record/", voice_views.voice_record_page, name="voice_record_alt"),
    path("analyze/", voice_views.voice_analyze_api, name="voice_analyze_api_alt"),
    path("results/", voice_views.voice_results_page, name="voice_results_alt"),
]
