"""Tests pour le processeur audio."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.multimodal.audio_processor import AudioProcessor
from src.multimodal.multimodal_types import (
    Language,
    VoiceGender,
    AudioAnalysis,
    SpeechSynthesis,
    SpeakerIdentification
)


@pytest.fixture
def audio_processor():
    """Fixture pour le processeur audio."""
    with patch('src.multimodal.audio_processor.AUDIO_LIBRARIES_AVAILABLE', True):
        with patch('src.multimodal.audio_processor.WhisperForConditionalGeneration') as mock_whisper_model, \
             patch('src.multimodal.audio_processor.WhisperProcessor') as mock_whisper_processor, \
             patch('src.multimodal.audio_processor.Wav2Vec2ForCTC') as mock_wav2vec_model, \
             patch('src.multimodal.audio_processor.Wav2Vec2Processor') as mock_wav2vec_processor:

            # Mock des modèles
            mock_whisper_model.from_pretrained.return_value = Mock()
            mock_whisper_processor.from_pretrained.return_value = Mock()
            mock_wav2vec_model.from_pretrained.return_value = Mock()
            mock_wav2vec_processor.from_pretrained.return_value = Mock()

            processor = AudioProcessor()
            yield processor


@pytest.fixture
def sample_audio():
    """Fixture pour un audio d'exemple."""
    # Crée un signal audio simple (1 seconde à 16kHz)
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # Tone à 440 Hz
    return audio.astype(np.float32)


def test_audio_processor_initialization(audio_processor):
    """Teste l'initialisation du processeur audio."""
    assert audio_processor.device == "cpu"
    assert audio_processor.sample_rate == 16000
    assert audio_processor.whisper_model is not None
    assert audio_processor.wav2vec_model is not None


def test_transcribe_audio(audio_processor, tmp_path):
    """Teste la transcription audio."""
    audio_path = tmp_path / "test_audio.wav"

    # Crée un fichier audio temporaire simulé
    with patch('src.multimodal.audio_processor.librosa') as mock_librosa:
        mock_librosa.load.return_value = (np.random.randn(16000), 16000)

        # Mock Whisper
        with patch.object(audio_processor, '_whisper_transcribe') as mock_whisper:
            mock_whisper.return_value = ("Test transcription", 0.9, [])

            result = audio_processor.transcribe_audio(audio_path)

            assert isinstance(result, AudioAnalysis)
            assert result.transcription == "Test transcription"
            assert result.confidence == 0.9
            assert result.language == Language.ENGLISH


def test_synthesize_speech(audio_processor):
    """Teste la synthèse vocale."""
    text = "Hello world"

    with patch.object(audio_processor, 'tts_engine') as mock_engine:
        # Mock pyttsx3
        mock_engine.getProperty.return_value = 200  # Rate
        mock_engine.save_to_file = Mock()
        mock_engine.runAndWait = Mock()

        with patch.object(audio_processor, '_load_audio') as mock_load:
            mock_load.return_value = (np.random.randn(22050), 22050)

            voice_config = SpeechSynthesis(
                text=text,
                voice_gender=VoiceGender.NEUTRAL
            )

            result = audio_processor.synthesize_speech(text, voice_config)

            assert isinstance(result, SpeechSynthesis)
            assert result.text == text
            assert result.voice_gender == VoiceGender.NEUTRAL


def test_identify_speakers(audio_processor, tmp_path):
    """Teste l'identification des locuteurs."""
    audio_path = tmp_path / "test_audio.wav"

    with patch('src.multimodal.audio_processor.librosa') as mock_librosa:
        mock_librosa.load.return_value = (np.random.randn(16000), 16000)

        result = audio_processor.identify_speakers(audio_path)

        assert isinstance(result, SpeakerIdentification)
        assert result.speaker_count >= 1
        assert len(result.speakers) >= 1


def test_get_audio_embedding(audio_processor, tmp_path):
    """Teste l'extraction d'embeddings audio."""
    audio_path = tmp_path / "test_audio.wav"

    with patch('src.multimodal.audio_processor.librosa') as mock_librosa, \
         patch.object(audio_processor, 'wav2vec_processor') as mock_processor, \
         patch.object(audio_processor, 'wav2vec_model') as mock_model:

        mock_librosa.load.return_value = (np.random.randn(16000), 16000)

        # Mock Wav2Vec2
        mock_processor.return_value = {"input_values": torch.randn(1, 16000)}
        mock_model.return_value.last_hidden_state = torch.randn(1, 100, 768)

        result = audio_processor.get_audio_embedding(audio_path)

        assert isinstance(result, np.ndarray)
        assert result.shape == (768,)  # Dimension d'embedding


def test_whisper_transcribe(audio_processor, sample_audio):
    """Teste la transcription Whisper."""
    with patch.object(audio_processor, 'whisper_processor') as mock_processor, \
         patch.object(audio_processor, 'whisper_model') as mock_model:

        # Mock Whisper
        mock_processor.return_value.input_features = torch.randn(1, 80, 3000)
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 50257]])  # EOS token
        mock_processor.batch_decode.return_value = ["Test transcription"]

        transcription, confidence, segments = audio_processor._whisper_transcribe(
            sample_audio, 16000, Language.ENGLISH, "transcribe"
        )

        assert transcription == "Test transcription"
        assert confidence == 0.85
        assert len(segments) == 1


def test_wav2vec_transcribe(audio_processor, sample_audio):
    """Teste la transcription Wav2Vec2."""
    with patch.object(audio_processor, 'wav2vec_processor') as mock_processor, \
         patch.object(audio_processor, 'wav2vec_model') as mock_model:

        # Mock Wav2Vec2
        mock_processor.return_value = {"input_values": torch.randn(1, 16000)}
        mock_logits = torch.randn(1, 100, 32)  # 32 tokens vocab
        mock_model.return_value.logits = mock_logits
        mock_processor.batch_decode.return_value = ["Test transcription"]

        transcription, confidence = audio_processor._wav2vec_transcribe(
            sample_audio, 16000
        )

        assert transcription == "Test transcription"
        assert 0 <= confidence <= 1


def test_identify_speakers_simple(audio_processor, sample_audio):
    """Teste l'identification simplifiée des locuteurs."""
    result = audio_processor._identify_speakers_simple(sample_audio)

    assert isinstance(result, dict)
    assert "speaker_count" in result
    assert "speakers" in result
    assert result["speaker_count"] >= 1
    assert len(result["speakers"]) >= 1


def test_get_supported_languages(audio_processor):
    """Teste la récupération des langues supportées."""
    languages = audio_processor.get_supported_languages()

    assert isinstance(languages, list)
    assert len(languages) >= 1
    assert Language.ENGLISH in languages


def test_get_available_voices(audio_processor):
    """Teste la récupération des voix disponibles."""
    with patch.object(audio_processor, 'tts_engine') as mock_engine:
        # Mock des voix
        mock_voice1 = Mock()
        mock_voice1.id = "voice1"
        mock_voice1.name = "Voice 1"
        mock_voice1.gender = "Male"

        mock_voice2 = Mock()
        mock_voice2.id = "voice2"
        mock_voice2.name = "Voice 2"
        mock_voice2.gender = "Female"

        mock_engine.getProperty.return_value = [mock_voice1, mock_voice2]

        voices = audio_processor.get_available_voices()

        assert len(voices) == 2
        assert voices[0]["id"] == "voice1"
        assert voices[1]["gender"] == "Female"


def test_change_voice(audio_processor):
    """Teste le changement de voix."""
    with patch.object(audio_processor, 'tts_engine') as mock_engine:
        mock_engine.setProperty = Mock()

        success = audio_processor.change_voice("voice_id")

        assert success is True
        mock_engine.setProperty.assert_called_with('voice', "voice_id")


def test_error_handling(audio_processor):
    """Teste la gestion d'erreurs."""
    # Fichier audio inexistant
    with pytest.raises(FileNotFoundError):
        audio_processor.transcribe_audio("nonexistent.wav")

    # Modèles non disponibles
    audio_processor.whisper_model = None
    audio_processor.wav2vec_model = None

    with patch('src.multimodal.audio_processor.librosa') as mock_librosa:
        mock_librosa.load.return_value = (np.random.randn(16000), 16000)

        result = audio_processor.transcribe_audio("dummy.wav")
        assert "non disponibles" in result.transcription

    # TTS non disponible
    audio_processor.tts_engine = None
    audio_processor.tts_model = None

    with pytest.raises(RuntimeError):
        audio_processor.synthesize_speech("test")


def test_load_audio(audio_processor, tmp_path):
    """Teste le chargement d'audio."""
    audio_path = tmp_path / "test.wav"

    with patch('src.multimodal.audio_processor.librosa') as mock_librosa:
        expected_audio = np.random.randn(16000).astype(np.float32)
        mock_librosa.load.return_value = (expected_audio, 16000)

        audio, sample_rate = audio_processor._load_audio(audio_path)

        assert np.array_equal(audio, expected_audio)
        assert sample_rate == 16000