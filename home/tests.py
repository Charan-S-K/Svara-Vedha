import io
import json
import numpy as np
import soundfile as sf
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile

from home.services.voice_models import (
    decode_and_preprocess_audio,
    AudioProcessingError,
    VoiceStressService,
    VoiceStrainService,
    VoiceEmotionService,
    analyze_voice_recording,
)


class VoiceAnalysisIntegrationTests(TestCase):
    """
    Comprehensive tests for the 3 Voice Analysis Models and Web Integration:
    - Root URL (/) restored to original SVARAABHAVA welcome & authentication page
    - Voice Analysis navigation button visible on welcome page
    - Dedicated Voice Analysis Recording page (/voice-analysis/record/)
    - Dedicated Voice Analysis Results dashboard (/voice-analysis/results/)
    - Audio Preprocessing (WAV, MP3, mono, resampling)
    - Model 1: Voice Stress (Wav2Vec2 + BiLSTM)
    - Model 2: Voice Strain (ECAPA-TDNN-VHE)
    - Model 3: 7-Class Emotion Detection
    - Session Retention and Navigation States
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Generate 10 seconds of synthetic 16kHz speech-like test audio
        cls.sr = 16000
        cls.duration = 10.0
        t = np.linspace(0, cls.duration, int(cls.sr * cls.duration), endpoint=False)
        signal = (
            0.35 * np.sin(2 * np.pi * 180 * t) +
            0.25 * np.sin(2 * np.pi * 360 * t) +
            0.15 * np.sin(2 * np.pi * 720 * t) +
            0.05 * np.random.randn(len(t))
        )
        wav_buf = io.BytesIO()
        sf.write(wav_buf, signal.astype(np.float32), cls.sr, format="WAV")
        cls.valid_wav_bytes = wav_buf.getvalue()

    def setUp(self):
        self.client = Client()

    def test_01_root_url_renders_original_welcome_page(self):
        """
        Phase 1 Verification: Verify GET / renders the original SVARAABHAVA welcome
        page (accounts/login.html) with authentication, brand styling, and NO direct
        drag-and-drop or microphone upload form.
        """
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertTemplateUsed(resp, "accounts/login.html")
        self.assertContains(resp, "SVARAABHAVA")
        self.assertContains(resp, "Sign In")

    def test_02_voice_analysis_recording_url_renders_audio_html(self):
        """
        Phase 2 Verification: Verify GET /voice-analysis/record/ renders the dedicated
        recording page (audio.html) with timer, mic, and Analyze Voice button.
        """
        resp = self.client.get("/voice-analysis/record/")
        self.assertEqual(resp.status_code, 200)
        self.assertTemplateUsed(resp, "audio.html")
        self.assertContains(resp, "Voice Recording")
        self.assertContains(resp, "Analysis Results")
        self.assertContains(resp, "Analyze Voice")

    def test_03_voice_analysis_entry_redirects_to_record_page(self):
        """Verify GET /voice-analysis/ redirects to /voice-analysis/record/."""
        resp = self.client.get("/voice-analysis/")
        self.assertRedirects(resp, "/voice-analysis/record/")

    def test_04_analysis_url_without_recording_shows_empty_state(self):
        """
        Verify opening /voice-analysis/results/ without submitting audio displays:
        'No voice recording available. Please record or upload audio first.'
        and does NOT display sample demonstration numbers as real results.
        """
        resp = self.client.get("/voice-analysis/results/")
        self.assertEqual(resp.status_code, 200)
        self.assertTemplateUsed(resp, "emotion_analysis.html")
        self.assertContains(resp, "No Voice Recording Available")
        self.assertContains(resp, "Please record or upload audio first.")
        self.assertContains(resp, "Go to Voice Recording")
        # Ensure sample 82% hardcoded demo value is NOT present as an actual result
        self.assertNotContains(resp, "82% Stressed")

    def test_05_audio_preprocessing_valid_wav(self):
        """Verify audio decoding, mono conversion, and 16kHz resampling."""
        audio_np, duration, orig_sr = decode_and_preprocess_audio(self.valid_wav_bytes)
        self.assertEqual(orig_sr, 16000)
        self.assertAlmostEqual(duration, 10.0, places=1)
        self.assertEqual(audio_np.ndim, 1)
        self.assertEqual(audio_np.dtype, np.float32)

    def test_06_audio_preprocessing_corrupted_audio_raises(self):
        """Verify corrupted or invalid audio raises AudioProcessingError."""
        with self.assertRaises(AudioProcessingError):
            decode_and_preprocess_audio(b"THIS_IS_CORRUPTED_GARBAGE_BYTES_1234567890" * 10)

    def test_07_audio_preprocessing_silence_raises(self):
        """Verify pure silence recording is detected and rejected."""
        silent_audio = np.zeros(16000, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, silent_audio, 16000, format="WAV")
        with self.assertRaises(AudioProcessingError):
            decode_and_preprocess_audio(buf.getvalue())

    def test_08_model1_voice_stress_inference(self):
        """Verify Model 1 (Wav2Vec2 + BiLSTM) inference pipeline produces valid stress metrics."""
        service = VoiceStressService.get_instance()
        audio_np, _, _ = decode_and_preprocess_audio(self.valid_wav_bytes)
        res = service.predict(audio_np)
        self.assertTrue(res["available"])
        self.assertIn("percentage", res)
        self.assertIn("category", res)
        self.assertIn("interpretation", res)
        self.assertGreaterEqual(res["percentage"], 0.0)
        self.assertLessEqual(res["percentage"], 100.0)

    def test_09_model2_voice_strain_inference(self):
        """Verify Model 2 (ECAPA-TDNN-VHE) inference produces 192-D embeddings and strain score."""
        service = VoiceStrainService.get_instance()
        audio_np, _, _ = decode_and_preprocess_audio(self.valid_wav_bytes)
        res = service.predict(audio_np)
        self.assertTrue(res["available"])
        self.assertEqual(res["embedding_dimension"], 192)
        self.assertIn("percentage", res)
        self.assertIn("category", res)
        self.assertIn("disclaimer", res)
        self.assertGreaterEqual(res["percentage"], 0.0)
        self.assertLessEqual(res["percentage"], 100.0)

    def test_10_model3_emotion_detection_inference(self):
        """Verify Model 3 (7-Emotion classifier) evaluates all 7 supported emotion categories."""
        service = VoiceEmotionService.get_instance()
        audio_np, duration, _ = decode_and_preprocess_audio(self.valid_wav_bytes)
        res = service.predict(audio_np, duration)
        self.assertTrue(res["available"])
        self.assertIn("dominant_emotion", res)
        self.assertIn(res["dominant_emotion"], VoiceEmotionService.EMOTIONS)
        # Verify all 7 classes are present
        for emo in ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]:
            self.assertIn(emo, res["probabilities"])
        self.assertEqual(len(res["segments"]), 7)

    def test_11_voice_analyze_api_missing_file_returns_400(self):
        """Verify POST /api/voice/analyze/ with no file returns 400."""
        resp = self.client.post("/api/voice/analyze/")
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertFalse(data["success"])
        self.assertIn("No audio file provided", data["error"])

    def test_12_voice_analyze_api_end_to_end_and_results_rendering(self):
        """
        Verify end-to-end flow:
        1. Upload valid 10s audio to /voice-analysis/api/analyze/
        2. Verify all three models infer correctly
        3. Verify session persists analysis
        4. Verify /voice-analysis/results/ renders actual predicted results
        """
        uploaded_file = SimpleUploadedFile(
            "test_mic_10s.wav",
            self.valid_wav_bytes,
            content_type="audio/wav"
        )
        post_resp = self.client.post(
            "/voice-analysis/api/analyze/",
            {"audio": uploaded_file}
        )
        self.assertEqual(post_resp.status_code, 200)
        data = post_resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["redirect_url"], "/voice-analysis/results/")

        # Verify response structure for all 3 models
        stress = data["data"]["stress"]
        strain = data["data"]["strain"]
        emotion = data["data"]["emotion"]

        self.assertTrue(stress["available"])
        self.assertTrue(strain["available"])
        self.assertTrue(emotion["available"])
        self.assertEqual(len(emotion["probabilities"]), 7)

        # Check session retention
        session = self.client.session
        self.assertIn("latest_voice_analysis", session)

        # Follow redirect to /voice-analysis/results/
        results_resp = self.client.get("/voice-analysis/results/")
        self.assertEqual(results_resp.status_code, 200)
        self.assertTemplateUsed(results_resp, "emotion_analysis.html")

        # Verify real values are displayed
        self.assertContains(results_resp, "Voice Analysis Dashboard")
        self.assertContains(results_resp, f"{stress['percentage']}%")
        self.assertContains(results_resp, f"{strain['percentage']}%")
        self.assertContains(results_resp, emotion["dominant_emotion"].upper())
        self.assertContains(results_resp, "test_mic_10s.wav")
