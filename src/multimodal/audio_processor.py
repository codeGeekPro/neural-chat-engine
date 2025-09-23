"""Processeur audio pour speech-to-text et text-to-speech."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np

# Imports conditionnels pour les bibliothèques audio
try:
    import torch
    from transformers import (
        WhisperProcessor, WhisperForConditionalGeneration,
        Wav2Vec2Processor, Wav2Vec2ForCTC,
        SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    )
    import librosa
    AUDIO_LIBRARIES_AVAILABLE = True
except ImportError:
    AUDIO_LIBRARIES_AVAILABLE = False
    torch = None
    WhisperProcessor = None
    WhisperForConditionalGeneration = None
    Wav2Vec2Processor = None
    Wav2Vec2ForCTC = None
    SpeechT5Processor = None
    SpeechT5ForTextToSpeech = None
    SpeechT5HifiGan = None
    librosa = None

try:
    import pyttsx3
    TTS_FALLBACK_AVAILABLE = True
except ImportError:
    TTS_FALLBACK_AVAILABLE = False
    pyttsx3 = None

from .multimodal_types import (
    AudioModelType,
    VoiceGender,
    Language,
    AudioAnalysis,
    SpeechSynthesis,
    SpeakerIdentification,
    AudioInput,
    AudioEmbedding
)


logger = logging.getLogger(__name__)


class AudioProcessor:
    """Processeur audio pour speech-to-text et text-to-speech."""

    def __init__(
        self,
        audio_model: str = "openai/whisper-base",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        sample_rate: int = 16000
    ):
        """Initialise le processeur audio.

        Args:
            audio_model: Modèle audio à utiliser
            device: Périphérique de calcul
            cache_dir: Répertoire de cache pour les modèles
            sample_rate: Taux d'échantillonnage audio
        """
        if not AUDIO_LIBRARIES_AVAILABLE:
            logger.warning(
                "Bibliothèques audio non disponibles. Installez transformers, "
                "torch, et librosa pour toutes les fonctionnalités."
            )

        self.audio_model_name = audio_model
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.sample_rate = sample_rate

        # Modèles
        self.whisper_model = None
        self.whisper_processor = None
        self.wav2vec_model = None
        self.wav2vec_processor = None
        self.tts_model = None
        self.tts_processor = None
        self.tts_vocoder = None

        # TTS de secours
        self.tts_engine = None

        # Cache des voix
        self.voice_cache: Dict[str, Any] = {}

        # Initialisation des modèles
        self._initialize_models()

        logger.info(f"AudioProcessor initialisé avec {audio_model}")

    def _initialize_models(self) -> None:
        """Initialise les modèles audio."""
        cache_dir = str(self.cache_dir) if self.cache_dir else None

        # Whisper pour speech-to-text
        try:
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                self.audio_model_name,
                cache_dir=cache_dir
            ).to(self.device)
            self.whisper_processor = WhisperProcessor.from_pretrained(
                self.audio_model_name,
                cache_dir=cache_dir
            )
            logger.info("Modèle Whisper chargé")
        except Exception as e:
            logger.warning(f"Impossible de charger Whisper: {e}")

        # Wav2Vec2 pour reconnaissance alternative
        try:
            wav2vec_model_name = "facebook/wav2vec2-base-960h"
            self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained(
                wav2vec_model_name,
                cache_dir=cache_dir
            ).to(self.device)
            self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(
                wav2vec_model_name,
                cache_dir=cache_dir
            )
            logger.info("Modèle Wav2Vec2 chargé")
        except Exception as e:
            logger.warning(f"Impossible de charger Wav2Vec2: {e}")

        # SpeechT5 pour text-to-speech
        try:
            tts_model_name = "microsoft/speecht5_tts"
            self.tts_model = SpeechT5ForTextToSpeech.from_pretrained(
                tts_model_name,
                cache_dir=cache_dir
            ).to(self.device)
            self.tts_processor = SpeechT5Processor.from_pretrained(
                tts_model_name,
                cache_dir=cache_dir
            )

            # Vocoder pour la synthèse
            vocoder_name = "microsoft/speecht5_hifigan"
            self.tts_vocoder = SpeechT5HifiGan.from_pretrained(
                vocoder_name,
                cache_dir=cache_dir
            ).to(self.device)

            logger.info("Modèle SpeechT5 chargé")
        except Exception as e:
            logger.warning(f"Impossible de charger SpeechT5: {e}")

        # TTS de secours
        if TTS_FALLBACK_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                logger.info("TTS de secours initialisé")
            except Exception as e:
                logger.warning(f"Impossible d'initialiser TTS de secours: {e}")

    def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        language: Language = Language.ENGLISH,
        task: str = "transcribe"
    ) -> AudioAnalysis:
        """Transcrit un fichier audio en texte.

        Args:
            audio_path: Chemin vers le fichier audio
            language: Langue de l'audio
            task: Tâche ("transcribe" ou "translate")

        Returns:
            Analyse audio avec transcription
        """
        start_time = time.time()
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Fichier audio non trouvé: {audio_path}")

        # Charge l'audio
        audio_array, sample_rate = self._load_audio(audio_path)

        # Utilise Whisper si disponible
        if self.whisper_model is not None:
            transcription, confidence, segments = self._whisper_transcribe(
                audio_array, sample_rate, language, task
            )
        elif self.wav2vec_model is not None:
            transcription, confidence = self._wav2vec_transcribe(audio_array, sample_rate)
            segments = []
        else:
            transcription = "Modèles de transcription non disponibles"
            confidence = 0.0
            segments = []

        # Analyse des locuteurs (simplifiée)
        speaker_analysis = self._identify_speakers_simple(audio_array)

        processing_time = time.time() - start_time

        return AudioAnalysis(
            audio_path=audio_path,
            transcription=transcription,
            language=language,
            speaker_count=speaker_analysis["speaker_count"],
            speakers_identified=speaker_analysis["speakers"],
            duration=len(audio_array) / sample_rate,
            confidence=confidence,
            segments=segments,
            processing_time=processing_time
        )

    def synthesize_speech(
        self,
        text: str,
        voice_config: Optional[SpeechSynthesis] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> SpeechSynthesis:
        """Synthétise la parole à partir de texte.

        Args:
            text: Texte à synthétiser
            voice_config: Configuration vocale
            output_path: Chemin de sortie pour l'audio

        Returns:
            Configuration de synthèse avec métadonnées
        """
        start_time = time.time()

        if voice_config is None:
            voice_config = SpeechSynthesis(text=text)

        # Utilise SpeechT5 si disponible
        if self.tts_model is not None and self.tts_processor is not None:
            audio_array = self._speecht5_synthesize(
                text, voice_config
            )
        elif self.tts_engine is not None:
            audio_array = self._fallback_tts_synthesize(text, voice_config)
        else:
            raise RuntimeError("Aucun système TTS disponible")

        # Sauvegarde si demandé
        if output_path:
            output_path = Path(output_path)
            self._save_audio(audio_array, output_path)

        processing_time = time.time() - start_time

        return SpeechSynthesis(
            text=text,
            voice_gender=voice_config.voice_gender,
            language=voice_config.language,
            speaking_rate=voice_config.speaking_rate,
            pitch=voice_config.pitch,
            volume=voice_config.volume,
            voice_name=voice_config.voice_name,
            output_path=output_path,
            processing_time=processing_time
        )

    def identify_speakers(
        self,
        audio_path: Union[str, Path],
        num_speakers: Optional[int] = None
    ) -> SpeakerIdentification:
        """Identifie les locuteurs dans un fichier audio.

        Args:
            audio_path: Chemin vers le fichier audio
            num_speakers: Nombre de locuteurs estimé

        Returns:
            Identification des locuteurs
        """
        start_time = time.time()
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Fichier audio non trouvé: {audio_path}")

        # Charge l'audio
        audio_array, sample_rate = self._load_audio(audio_path)

        # Analyse simplifiée des locuteurs
        # En production, utiliser pyannote.audio ou similaires
        speaker_analysis = self._identify_speakers_simple(audio_array)

        processing_time = time.time() - start_time

        return SpeakerIdentification(
            speakers=speaker_analysis["speakers"],
            speaker_count=speaker_analysis["speaker_count"],
            confidence_scores=speaker_analysis.get("confidence_scores", {}),
            processing_time=processing_time
        )

    def get_audio_embedding(self, audio_path: Union[str, Path]) -> AudioEmbedding:
        """Extrait l'embedding d'un fichier audio."""
        if self.wav2vec_model is None:
            raise RuntimeError("Modèle Wav2Vec2 non disponible pour les embeddings")

        # Charge l'audio
        audio_array, sample_rate = self._load_audio(audio_path)

        # Prétraitement
        inputs = self.wav2vec_processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.wav2vec_model(**inputs)

        # Utilise les dernières couches cachées comme embedding
        embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.cpu().numpy().flatten()

    def _whisper_transcribe(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        language: Language,
        task: str
    ) -> Tuple[str, float, List[Dict[str, Any]]]:
        """Transcription avec Whisper."""
        # Prétraitement
        input_features = self.whisper_processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).input_features.to(self.device)

        # Génération des tokens
        predicted_ids = self.whisper_model.generate(
            input_features,
            language=language.value,
            task=task
        )

        # Décodage
        transcription = self.whisper_processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        # Confiance approximative (simplifiée)
        confidence = 0.85

        # Segments (simplifiés)
        segments = [{
            "text": transcription,
            "start": 0.0,
            "end": len(audio_array) / sample_rate,
            "confidence": confidence
        }]

        return transcription, confidence, segments

    def _wav2vec_transcribe(
        self,
        audio_array: np.ndarray,
        sample_rate: int
    ) -> Tuple[str, float]:
        """Transcription avec Wav2Vec2."""
        # Prétraitement
        inputs = self.wav2vec_processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.wav2vec_model(**inputs).logits

        # Prédiction des tokens
        predicted_ids = torch.argmax(logits, dim=-1)

        # Décodage
        transcription = self.wav2vec_processor.batch_decode(predicted_ids)[0]

        # Confiance
        probs = torch.softmax(logits, dim=-1)
        confidence = torch.max(probs, dim=-1)[0].mean().item()

        return transcription, confidence

    def _speecht5_synthesize(
        self,
        text: str,
        voice_config: SpeechSynthesis
    ) -> np.ndarray:
        """Synthèse avec SpeechT5."""
        # Tokenisation du texte
        inputs = self.tts_processor(text=text, return_tensors="pt").to(self.device)

        # Embedding de locuteur (simplifié - utiliser un vrai embedding en production)
        speaker_embeddings = torch.randn(1, 512).to(self.device)

        with torch.no_grad():
            # Génération du spectrogramme
            spectrogram = self.tts_model.generate_speech(
                inputs["input_ids"],
                speaker_embeddings,
                vocoder=self.tts_vocoder
            )

        # Conversion en waveform
        audio_array = spectrogram.cpu().numpy().flatten()

        return audio_array

    def _fallback_tts_synthesize(
        self,
        text: str,
        voice_config: SpeechSynthesis
    ) -> np.ndarray:
        """Synthèse TTS de secours avec pyttsx3."""
        # Configure la voix
        voices = self.tts_engine.getProperty('voices')

        # Sélection de la voix selon le genre
        if voice_config.voice_gender == VoiceGender.FEMALE:
            female_voices = [v for v in voices if v.gender and 'female' in v.gender.lower()]
            if female_voices:
                self.tts_engine.setProperty('voice', female_voices[0].id)
        elif voice_config.voice_gender == VoiceGender.MALE:
            male_voices = [v for v in voices if v.gender and 'male' in v.gender.lower()]
            if male_voices:
                self.tts_engine.setProperty('voice', male_voices[0].id)

        # Configure la vitesse
        rate = self.tts_engine.getProperty('rate')
        self.tts_engine.setProperty('rate', int(rate * voice_config.speaking_rate))

        # Configure le volume
        self.tts_engine.setProperty('volume', voice_config.volume)

        # Sauvegarde temporaire
        temp_file = Path("temp_tts.wav")
        self.tts_engine.save_to_file(text, str(temp_file))
        self.tts_engine.runAndWait()

        # Charge le fichier audio généré
        audio_array, _ = self._load_audio(temp_file)

        # Nettoie
        temp_file.unlink(missing_ok=True)

        return audio_array

    def _identify_speakers_simple(
        self,
        audio_array: np.ndarray
    ) -> Dict[str, Any]:
        """Identification simplifiée des locuteurs."""
        # Analyse basique de l'énergie et des silences
        # En production, utiliser un vrai système de diarization

        # Calcul de l'énergie
        energy = np.sum(audio_array ** 2)

        # Seuils simples pour estimer le nombre de locuteurs
        duration = len(audio_array) / 16000  # Approximation

        if duration < 10:  # Court
            speaker_count = 1
        elif duration < 60:  # Moyen
            speaker_count = min(3, max(1, int(duration / 15)))
        else:  # Long
            speaker_count = min(5, max(2, int(duration / 20)))

        # Locuteurs fictifs pour la démonstration
        speakers = []
        for i in range(speaker_count):
            speakers.append({
                "id": f"speaker_{i+1}",
                "segments": [{"start": i * duration / speaker_count,
                            "end": (i+1) * duration / speaker_count}],
                "confidence": 0.7
            })

        return {
            "speaker_count": speaker_count,
            "speakers": speakers,
            "confidence_scores": {f"speaker_{i+1}": 0.7 for i in range(speaker_count)}
        }

    def _load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Charge un fichier audio."""
        if librosa is None:
            raise ImportError("librosa non disponible pour le chargement audio")

        audio_array, sample_rate = librosa.load(audio_path, sr=self.sample_rate)
        return audio_array, sample_rate

    def _save_audio(
        self,
        audio_array: np.ndarray,
        output_path: Union[str, Path],
        sample_rate: int = 22050
    ) -> None:
        """Sauvegarde un array audio dans un fichier."""
        if librosa is None:
            raise ImportError("librosa non disponible pour la sauvegarde audio")

        # Normalise l'audio
        audio_array = audio_array / np.max(np.abs(audio_array))

        librosa.output.write_wav(str(output_path), audio_array, sample_rate)

    def get_supported_languages(self) -> List[Language]:
        """Retourne les langues supportées."""
        return [Language.ENGLISH, Language.FRENCH, Language.SPANISH]

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Retourne les voix disponibles."""
        voices = []

        if self.tts_engine:
            engine_voices = self.tts_engine.getProperty('voices')
            for voice in engine_voices:
                voices.append({
                    "id": voice.id,
                    "name": voice.name,
                    "gender": getattr(voice, 'gender', 'unknown'),
                    "language": getattr(voice, 'languages', ['unknown'])[0] if hasattr(voice, 'languages') else 'unknown'
                })

        return voices

    def change_voice(
        self,
        voice_id: str,
        gender: Optional[VoiceGender] = None,
        language: Optional[Language] = None
    ) -> bool:
        """Change la voix actuelle."""
        if self.tts_engine:
            try:
                self.tts_engine.setProperty('voice', voice_id)
                return True
            except Exception as e:
                logger.error(f"Erreur lors du changement de voix: {e}")
                return False
        return False