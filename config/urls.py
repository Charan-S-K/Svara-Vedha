"""
URL configuration for SVARAABHAVA / VoiceStressAI project.
"""
from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect
from home import voice_views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("allauth.urls")),

    # =========================================================================
    # DEDICATED VOICE ANALYSIS FEATURE ROUTES (Phase 2 Feature Architecture)
    # =========================================================================
    path("voice-analysis/", lambda request: redirect("/voice-analysis/record/"), name="voice_analysis_entry"),
    path("voice-analysis/record/", voice_views.voice_record_page, name="voice_record"),
    path("voice-analysis/results/", voice_views.voice_results_page, name="voice_results"),
    path("voice-analysis/api/analyze/", voice_views.voice_analyze_api, name="voice_analyze_api_feature"),

    # Direct API and aliases for backward-compatibility
    path("api/voice/analyze/", voice_views.voice_analyze_api, name="voice_analyze_api"),
    path("audio/", voice_views.voice_record_page, name="audio"),
    path("analysis/", voice_views.voice_results_page, name="analysis_alias"),

    # =========================================================================
    # RESTORED ORIGINAL PROJECT ARCHITECTURE (Phase 1)
    # Original welcome & login page at root '/', authentication, biometrics & dashboard
    # =========================================================================
    path("", include("accounts.urls")),
    path("dashboard/", include("home.urls")),
    path("voice/", include("home.voice_urls")),
]
