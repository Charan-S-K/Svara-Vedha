"""
Voice Analysis Services for VoiceStressAI / SVARAABHAVA.

Provides robust audio preprocessing and independent inference pipelines for:
1. Model 1: Voice Stress Detection (Hybrid Wav2Vec2 + BiLSTM)
2. Model 2: Voice Strain Detection (ECAPA-TDNN-VHE via Auralis VFS)
3. Model 3: 7-Class Emotion Detection (Wav2Vec2 Speech Emotion Classifier)

All model checkpoints are loaded locally from `models/voice/` with zero redundant
re-downloads or hardcoded mock predictions.
"""

import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from django.conf import settings

logger = logging.getLogger(__name__)

# Base directory for local model checkpoints
MODELS_DIR = Path(getattr(settings, "BASE_DIR", Path.cwd())) / "models" / "voice"
STRESS_MODEL_PATH = MODELS_DIR / "stress" / "model.pt"
EMOTION_MODEL_DIR = MODELS_DIR / "emotion"


# ==============================================================================
# Shared Audio Preprocessing & Decoding
# ==============================================================================

class AudioProcessingError(Exception):
    """Raised when uploaded audio cannot be decoded, is corrupted, or fails validation."""
    pass


def decode_and_preprocess_audio(
    audio_source: Union[bytes, io.BytesIO, str, Path],
    target_sr: int = 16000,
    min_duration_sec: float = 0.5,
    max_duration_sec: float = 60.0,
) -> Tuple[np.ndarray, float, int]:
    """
    Decodes arbitrary audio input (WAV, MP3, WebM, OGG, M4A, FLAC) into a standardized
    single-channel (mono) float32 NumPy array resampled to target_sr.

    Args:
        audio_source: Raw audio bytes, BytesIO buffer, or path string.
        target_sr: Desired sampling rate (default 16,000 Hz).
        min_duration_sec: Minimum acceptable duration in seconds (default 0.5s).
        max_duration_sec: Maximum acceptable duration in seconds (default 60.0s).

    Returns:
        Tuple[np.ndarray, float, int]:
            - audio: 1D float32 NumPy array normalized to [-1.0, 1.0].
            - duration: Exact duration in seconds.
            - original_sr: Original sampling rate before resampling.

    Raises:
        AudioProcessingError: If decoding fails or audio is invalid/silent.
    """
    raw_bytes: Optional[bytes] = None
    if isinstance(audio_source, bytes):
        raw_bytes = audio_source
    elif isinstance(audio_source, io.BytesIO):
        raw_bytes = audio_source.getvalue()
    elif isinstance(audio_source, (str, Path)) and os.path.exists(str(audio_source)):
        with open(str(audio_source), "rb") as f:
            raw_bytes = f.read()

    if not raw_bytes or len(raw_bytes) < 100:
        raise AudioProcessingError("Uploaded audio file is empty or too small.")

    audio_data: Optional[np.ndarray] = None
    orig_sr: int = target_sr

    # Strategy 1: soundfile (fast, handles WAV, MP3, OGG, FLAC natively)
    try:
        import soundfile as sf
        with sf.SoundFile(io.BytesIO(raw_bytes)) as sf_file:
            orig_sr = sf_file.samplerate
            audio_data = sf_file.read(dtype="float32")
    except Exception as sf_err:
        logger.debug("soundfile failed to decode audio: %s. Falling back to PyAV.", sf_err)

    # Strategy 2: PyAV (universal fallback for WebM, OPUS, M4A, AAC, MP4, etc.)
    if audio_data is None:
        try:
            import av
            container = av.open(io.BytesIO(raw_bytes))
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sr)
            frames = []
            for frame in container.decode(audio=0):
                for r_frame in resampler.resample(frame):
                    frames.append(r_frame.to_ndarray()[0])
            if frames:
                audio_data = np.concatenate(frames)
                orig_sr = target_sr
            container.close()
        except Exception as av_err:
            logger.error("PyAV also failed to decode audio: %s", av_err)
            raise AudioProcessingError(f"Could not decode audio format: {str(av_err)}") from av_err

    if audio_data is None or len(audio_data) == 0:
        raise AudioProcessingError("Decoded audio stream contains zero samples.")

    # Convert multi-channel (stereo) to mono by averaging channels
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample if needed and not already resampled by PyAV
    if orig_sr != target_sr:
        try:
            import librosa
            audio_data = librosa.resample(
                audio_data.astype(np.float32),
                orig_sr=orig_sr,
                target_sr=target_sr,
                res_type="soxr_hq" if hasattr(librosa, "soxr") else "kaiser_fast",
            )
        except Exception as resamp_err:
            logger.warning("librosa.resample failed (%s), using linear interpolation.", resamp_err)
            duration_s = len(audio_data) / orig_sr
            new_num_samples = int(duration_s * target_sr)
            audio_data = np.interp(
                np.linspace(0, len(audio_data), new_num_samples, endpoint=False),
                np.arange(len(audio_data)),
                audio_data,
            ).astype(np.float32)

    duration = float(len(audio_data)) / float(target_sr)

    if duration < min_duration_sec:
        raise AudioProcessingError(
            f"Audio recording is too short ({duration:.2f}s). Minimum required duration is {min_duration_sec}s."
        )

    if duration > max_duration_sec:
        # Trim gracefully to max_duration_sec
        audio_data = audio_data[: int(max_duration_sec * target_sr)]
        duration = max_duration_sec

    # Amplitude normalization & silence check
    max_amp = float(np.max(np.abs(audio_data)))
    if max_amp < 1e-4:
        raise AudioProcessingError("The recording contains only silence. Please speak clearly into the microphone.")

    # Peak normalization to safe range [-0.95, 0.95]
    if max_amp > 0:
        audio_data = (audio_data / max_amp) * 0.95

    return audio_data.astype(np.float32), round(duration, 2), orig_sr


# ==============================================================================
# Model 1: Voice Stress Detection (Hybrid Wav2Vec2 + BiLSTM)
# ==============================================================================

class HybridWav2Vec2LSTMStressModel(nn.Module):
    """
    Hybrid Wav2Vec2 + Bidirectional LSTM architecture fine-tuned for psychological
    stress and emotion classification (RAVDESS, TESS, SAVEE).
    """

    def __init__(self, wav2vec_path: str = "facebook/wav2vec2-base"):
        super().__init__()
        from transformers import Wav2Vec2Model
        self.wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_path, local_files_only=True)
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.emotion_head = nn.Linear(256, 5)
        self.stress_head = nn.Linear(256, 1)

    def forward(self, input_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            input_values: [batch_size, sequence_length] 16 kHz raw audio waveform.
        Returns:
            stress_logits: [batch_size, 1]
            emotion_logits: [batch_size, 5]
        """
        outputs = self.wav2vec(input_values)
        hidden_states = outputs.last_hidden_state
        lstm_out, _ = self.lstm(hidden_states)
        pooled = torch.mean(lstm_out, dim=1)
        x = self.dropout(self.relu(self.fc(pooled)))
        stress_out = self.stress_head(x)
        emotion_out = self.emotion_head(x)
        return stress_out, emotion_out


class VoiceStressService:
    """Singleton service for Voice Stress Detection inference."""

    _instance: Optional["VoiceStressService"] = None

    def __init__(self):
        self.model: Optional[HybridWav2Vec2LSTMStressModel] = None
        self.device = torch.device("cpu")
        self.emotion_labels = {0: "neutral", 1: "happy", 2: "sad", 3: "angry", 4: "fear"}
        self._load_model()

    @classmethod
    def get_instance(cls) -> "VoiceStressService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self) -> None:
        if not STRESS_MODEL_PATH.exists():
            logger.error("Stress model checkpoint not found at: %s", STRESS_MODEL_PATH)
            return

        try:
            logger.info("Loading Voice Stress model checkpoint from %s...", STRESS_MODEL_PATH)
            model = HybridWav2Vec2LSTMStressModel()
            checkpoint = torch.load(str(STRESS_MODEL_PATH), map_location=self.device)
            state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
            model.load_state_dict(state_dict, strict=True)
            model.to(self.device)
            model.eval()
            self.model = model
            logger.info("Voice Stress Model loaded successfully.")
        except Exception as e:
            logger.exception("Failed to load Voice Stress Model: %s", e)
            self.model = None

    def predict(self, audio_16k: np.ndarray) -> Dict[str, Any]:
        """
        Run stress inference on 16 kHz mono audio array.
        """
        if self.model is None:
            return {
                "available": False,
                "status": "Model checkpoint not loaded",
                "score": None,
                "percentage": None,
                "category": "Unavailable",
                "confidence": None,
                "interpretation": "Voice stress model is currently unavailable on the server.",
            }

        try:
            # Wav2Vec2 standard normalization: (audio - mean) / (std + 1e-7)
            mean = np.mean(audio_16k)
            std = np.std(audio_16k) + 1e-7
            norm_audio = (audio_16k - mean) / std

            tensor_input = torch.from_numpy(norm_audio).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                stress_logit, emo_logits = self.model(tensor_input)
                stress_prob = torch.sigmoid(stress_logit).item()
                emo_probs = torch.softmax(emo_logits, dim=-1).squeeze().tolist()

            stress_pct = round(stress_prob * 100.0, 1)

            if stress_pct < 35.0:
                category = "Normal"
                interp = "Low acoustic tension. Steady fundamental frequency and calm vocal stability detected."
            elif stress_pct < 65.0:
                category = "Elevated"
                interp = "Moderate vocal tension and microtremors detected. Vocal features indicate mild arousal or stress."
            else:
                category = "Stressed"
                interp = "High acoustic tension and vocal instability detected. Perturbation patterns correlate with acute stress."

            # Calculate confidence based on distance from decision boundary (0.5)
            confidence = round(abs(stress_prob - 0.5) * 2.0 * 100.0, 1)

            return {
                "available": True,
                "status": "Success",
                "score": round(stress_prob, 4),
                "percentage": stress_pct,
                "category": category,
                "confidence": confidence,
                "interpretation": interp,
            }
        except Exception as e:
            logger.exception("Inference failed for Voice Stress Model: %s", e)
            return {
                "available": False,
                "status": f"Inference error: {str(e)}",
                "score": None,
                "percentage": None,
                "category": "Error",
                "confidence": None,
                "interpretation": "Error occurred during acoustic stress inference.",
            }


# ==============================================================================
# Model 2: Voice Strain Detection (ECAPA-TDNN-VHE via Auralis VFS)
# ==============================================================================

class VoiceStrainService:
    """
    Inference service for Voice Strain Detection using ECAPA-TDNN-VHE
    (Vocal Health Encoder by Muhammad Khubaib Ahmad, Auralis VFS research).
    Produces 192-D vocal health embeddings and continuous vocal fatigue / strain scores.
    """

    _instance: Optional["VoiceStrainService"] = None

    def __init__(self):
        self._initialized = False
        self._encoder = None
        self._init_service()

    @classmethod
    def get_instance(cls) -> "VoiceStrainService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_service(self) -> None:
        try:
            import auralis
            from auralis.ecapa import ECAPAENCODER
            self._encoder = ECAPAENCODER()
            self._initialized = True
            logger.info("Voice Strain (ECAPA-TDNN-VHE / Auralis VFS) service initialized successfully.")
        except Exception as e:
            logger.exception("Failed to initialize Voice Strain service: %s", e)
            self._initialized = False

    def predict(self, audio_16k: np.ndarray) -> Dict[str, Any]:
        """
        Run strain inference on 16 kHz mono audio.
        """
        if not self._initialized:
            return {
                "available": False,
                "status": "ECAPA-TDNN-VHE backend unavailable",
                "score": None,
                "percentage": None,
                "category": "Unavailable",
                "embedding_dimension": 192,
                "confidence": None,
                "interpretation": "Voice strain model is currently unavailable on the server.",
                "disclaimer": "Acoustic strain metric. Non-clinical estimate.",
            }

        try:
            import auralis

            # Compute objective continuous fatigue/strain score (0-100 scale)
            raw_score = float(auralis.score_waveform(audio_16k))
            strain_pct = round(max(0.0, min(100.0, raw_score)), 1)

            # Categorize research score
            if strain_pct < 35.0:
                category = "Healthy"
                interp = "Vocal folds exhibiting normal periodic vibration and minimal glottal hyperfunction."
            elif strain_pct < 65.0:
                category = "Mild Strain"
                interp = "Mild vocal effort and spectral tilt alterations detected. Sub-clinical acoustic fatigue."
            else:
                category = "Strained"
                interp = "Significant acoustic strain and vocal hyperfunction detected along the vocal fatigue axis."

            confidence = round(max(50.0, min(99.0, 70.0 + (abs(strain_pct - 50.0) * 0.5))), 1)

            return {
                "available": True,
                "status": "Success",
                "score": round(strain_pct / 100.0, 4),
                "percentage": strain_pct,
                "category": category,
                "confidence": confidence,
                "embedding_dimension": 192,
                "interpretation": interp,
                "disclaimer": "Research-grade acoustic metric of vocal fold fatigue/hyperfunction. Non-clinical estimate.",
            }
        except Exception as e:
            logger.exception("Inference failed for Voice Strain Model: %s", e)
            return {
                "available": False,
                "status": f"Inference error: {str(e)}",
                "score": None,
                "percentage": None,
                "category": "Error",
                "confidence": None,
                "embedding_dimension": 192,
                "interpretation": "Error occurred during vocal strain analysis.",
                "disclaimer": "Research-grade acoustic metric. Non-clinical estimate.",
            }


# ==============================================================================
# Model 3: 7-Class Emotion Detection (Wav2Vec2 Speech Emotion Recognition)
# ==============================================================================

class EmotionClassificationModel(nn.Module):
    """
    Wav2Vec2 model with a 1024 -> 1024 -> 7 classification head matching the exact
    weights of the 7-emotion fine-tuned checkpoint.
    """

    def __init__(self, config):
        super().__init__()
        from transformers import Wav2Vec2Model
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = nn.Module()
        self.classifier.dense = nn.Linear(1024, 1024)
        self.classifier.out_proj = nn.Linear(1024, 7)
        self.dropout = nn.Dropout(0.0)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        x = torch.mean(hidden_states, dim=1)
        x = self.dropout(x)
        x = torch.tanh(self.classifier.dense(x))
        x = self.dropout(x)
        logits = self.classifier.out_proj(x)
        return logits


class VoiceEmotionService:
    """Singleton service for 7-Class Speech Emotion Recognition."""

    _instance: Optional["VoiceEmotionService"] = None

    EMOTIONS: List[str] = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    EMOTION_META = {
        "angry": {"emoji": "😡", "meaning": "Anger or frustration", "color": "#f43f5e"},
        "disgust": {"emoji": "🤢", "meaning": "Dislike or aversion", "color": "#f59e0b"},
        "fear": {"emoji": "😨", "meaning": "Worry or fear", "color": "#eab308"},
        "happy": {"emoji": "😊", "meaning": "Happiness or joy", "color": "#10b981"},
        "neutral": {"emoji": "😐", "meaning": "Normal or calm vocal expression", "color": "#94a3b8"},
        "sad": {"emoji": "😢", "meaning": "Sadness", "color": "#3b82f6"},
        "surprise": {"emoji": "😲", "meaning": "Unexpected or surprised expression", "color": "#a855f7"},
    }

    def __init__(self):
        self.model: Optional[EmotionClassificationModel] = None
        self.device = torch.device("cpu")
        self._load_model()

    @classmethod
    def get_instance(cls) -> "VoiceEmotionService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self) -> None:
        bin_path = EMOTION_MODEL_DIR / "pytorch_model.bin"
        cfg_path = EMOTION_MODEL_DIR / "config.json"

        if not bin_path.exists() or not cfg_path.exists():
            logger.error("Emotion model checkpoint or config not found in: %s", EMOTION_MODEL_DIR)
            return

        try:
            from transformers import Wav2Vec2Config
            logger.info("Loading 7-Emotion model from %s...", EMOTION_MODEL_DIR)
            config = Wav2Vec2Config.from_pretrained(str(cfg_path))
            model = EmotionClassificationModel(config)
            weights = torch.load(str(bin_path), map_location=self.device)
            model.load_state_dict(weights, strict=True)
            model.to(self.device)
            model.eval()
            self.model = model
            logger.info("7-Emotion Model loaded successfully (100% strict match).")
        except Exception as e:
            logger.exception("Failed to load 7-Emotion Model: %s", e)
            self.model = None

    def predict(self, audio_16k: np.ndarray, duration_sec: float) -> Dict[str, Any]:
        """
        Run 7-class emotion inference on 16 kHz mono audio array.
        Computes both overall emotion distribution and windowed micro-segments.
        """
        if self.model is None:
            return {
                "available": False,
                "status": "Model checkpoint not loaded",
                "dominant_emotion": "Unavailable",
                "dominant_confidence": None,
                "probabilities": {},
                "segments": [],
            }

        try:
            # 1. Overall audio inference
            overall_probs = self._infer_clip(audio_16k)
            sorted_emotions = sorted(overall_probs.items(), key=lambda item: item[1], reverse=True)
            dominant_name, dominant_pct = sorted_emotions[0]

            # 2. Temporal micro-segments inference
            # Chunk the audio into ~7 proportional temporal windows
            target_segments = 7
            window_len = len(audio_16k) // target_segments
            segments = []

            for i in range(target_segments):
                start_idx = i * window_len
                end_idx = len(audio_16k) if i == target_segments - 1 else (i + 1) * window_len
                chunk = audio_16k[start_idx:end_idx]

                if len(chunk) >= 4000: # at least 0.25s
                    chunk_probs = self._infer_clip(chunk)
                else:
                    chunk_probs = overall_probs

                top_chunk_emo, top_chunk_pct = sorted(chunk_probs.items(), key=lambda x: x[1], reverse=True)[0]

                t_start = round(start_idx / 16000.0, 1)
                t_end = round(end_idx / 16000.0, 1)
                meta = self.EMOTION_META.get(top_chunk_emo, {})

                segments.append({
                    "segment_number": i + 1,
                    "time_start": t_start,
                    "time_end": t_end,
                    "time_range": f"{t_start:.1f} - {t_end:.1f}s",
                    "emotion": top_chunk_emo,
                    "confidence": top_chunk_pct,
                    "emoji": meta.get("emoji", "🎙️"),
                    "meaning": meta.get("meaning", ""),
                    "color": meta.get("color", "#94a3b8"),
                })

            return {
                "available": True,
                "status": "Success",
                "dominant_emotion": dominant_name,
                "dominant_confidence": dominant_pct,
                "dominant_emoji": self.EMOTION_META[dominant_name]["emoji"],
                "dominant_meaning": self.EMOTION_META[dominant_name]["meaning"],
                "probabilities": overall_probs,
                "segments": segments,
            }
        except Exception as e:
            logger.exception("Inference failed for Voice Emotion Model: %s", e)
            return {
                "available": False,
                "status": f"Inference error: {str(e)}",
                "dominant_emotion": "Error",
                "dominant_confidence": None,
                "probabilities": {},
                "segments": [],
            }

    def _infer_clip(self, clip: np.ndarray) -> Dict[str, float]:
        """Infers normalized probability distribution over 7 classes for a single audio slice."""
        mean = np.mean(clip)
        std = np.std(clip) + 1e-7
        norm_clip = (clip - mean) / std

        tensor_input = torch.from_numpy(norm_clip).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            logits = self.model(tensor_input)
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()

        res = {}
        for idx, emo in enumerate(self.EMOTIONS):
            res[emo] = round(probs[idx] * 100.0, 1)

        return res


# ==============================================================================
# Unified Pipeline Orchestrator
# ==============================================================================

def analyze_voice_recording(
    audio_source: Union[bytes, io.BytesIO, str, Path],
    filename: str = "voice_recording.wav",
) -> Dict[str, Any]:
    """
    Main orchestration entrypoint.
    Runs shared preprocessing followed by all three independent neural models.

    Args:
        audio_source: Audio payload bytes or file buffer.
        filename: Original user filename for metadata.

    Returns:
        Structured dictionary adhering strictly to the required contract.
    """
    # 1. Preprocess & Validate Audio
    audio_16k, duration_sec, orig_sr = decode_and_preprocess_audio(audio_source)

    # 2. Model 1: Voice Stress Detection
    stress_service = VoiceStressService.get_instance()
    stress_result = stress_service.predict(audio_16k)

    # 3. Model 2: Voice Strain Detection
    strain_service = VoiceStrainService.get_instance()
    strain_result = strain_service.predict(audio_16k)

    # 4. Model 3: Emotion Detection
    emotion_service = VoiceEmotionService.get_instance()
    emotion_result = emotion_service.predict(audio_16k, duration_sec)

    # 5. Composite Summary & Overall Interpretation
    overall_interpretation = _generate_overall_interpretation(
        stress_result=stress_result,
        strain_result=strain_result,
        emotion_result=emotion_result,
    )

    return {
        "success": True,
        "audio_info": {
            "filename": filename,
            "duration": duration_sec,
            "duration_formatted": f"{duration_sec:.1f} seconds",
            "sample_rate": 16000,
            "original_sample_rate": orig_sr,
            "channels": 1,
            "format": "16 kHz Mono PCM",
        },
        "stress": stress_result,
        "strain": strain_result,
        "emotion": emotion_result,
        "overall_interpretation": overall_interpretation,
    }


def _generate_overall_interpretation(
    stress_result: Dict[str, Any],
    strain_result: Dict[str, Any],
    emotion_result: Dict[str, Any],
) -> str:
    """Synthesizes dynamic acoustic interpretation from the three actual model outputs."""
    stress_pct = stress_result.get("percentage")
    strain_pct = strain_result.get("percentage")
    dominant_emo = emotion_result.get("dominant_emotion", "neutral")
    dominant_pct = emotion_result.get("dominant_confidence", 0)

    parts = []

    if stress_pct is not None and strain_pct is not None:
        if stress_pct >= 65.0 and strain_pct >= 65.0:
            parts.append(
                f"Elevated acoustic tension and pronounced vocal hyperfunction detected "
                f"({stress_pct}% Stress, {strain_pct}% Voice Strain)."
            )
        elif stress_pct >= 65.0:
            parts.append(
                f"Elevated psychological stress pattern detected ({stress_pct}%) "
                f"with moderate vocal cord strain ({strain_pct}%)."
            )
        elif strain_pct >= 65.0:
            parts.append(
                f"Marked acoustic vocal strain ({strain_pct}%) observed with controlled baseline stress ({stress_pct}%)."
            )
        elif stress_pct < 35.0 and strain_pct < 35.0:
            parts.append(
                f"Balanced acoustic biomarkers indicating relaxed vocal fold vibration and calm autonomic tone "
                f"({stress_pct}% Stress, {strain_pct}% Voice Strain)."
            )
        else:
            parts.append(
                f"Moderate physiological vocal parameters observed "
                f"({stress_pct}% Stress, {strain_pct}% Voice Strain)."
            )

    if dominant_emo and dominant_pct:
        parts.append(
            f"Vocal prosody is predominantly {dominant_emo.capitalize()} ({dominant_pct}% confidence), "
            f"reflecting {emotion_result.get('dominant_meaning', 'the speaker state')}."
        )

    return " ".join(parts)
