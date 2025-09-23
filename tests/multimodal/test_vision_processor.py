"""Tests pour le processeur de vision."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.multimodal.vision_processor import VisionProcessor
from src.multimodal.multimodal_types import (
    DetailLevel,
    ImageAnalysis,
    VisualQuestion,
    OCRResult,
    ObjectDetection,
    SceneDescription
)


@pytest.fixture
def vision_processor():
    """Fixture pour le processeur de vision."""
    with patch('src.multimodal.vision_processor.VISION_LIBRARIES_AVAILABLE', True):
        with patch('src.multimodal.vision_processor.CLIPModel') as mock_clip_model, \
             patch('src.multimodal.vision_processor.CLIPProcessor') as mock_clip_processor, \
             patch('src.multimodal.vision_processor.BlipForConditionalGeneration') as mock_blip_model, \
             patch('src.multimodal.vision_processor.BlipProcessor') as mock_blip_processor:

            # Mock des modèles
            mock_clip_model.from_pretrained.return_value = Mock()
            mock_clip_processor.from_pretrained.return_value = Mock()
            mock_blip_model.from_pretrained.return_value = Mock()
            mock_blip_processor.from_pretrained.return_value = Mock()

            processor = VisionProcessor()
            yield processor


@pytest.fixture
def sample_image():
    """Fixture pour une image d'exemple."""
    # Crée une image RGB simple
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return image


def test_vision_processor_initialization(vision_processor):
    """Teste l'initialisation du processeur de vision."""
    assert vision_processor.device == "cpu"
    assert vision_processor.max_image_size == (224, 224)
    assert vision_processor.clip_model is not None
    assert vision_processor.blip_model is not None


def test_analyze_image(vision_processor, tmp_path):
    """Teste l'analyse d'image."""
    # Crée une image temporaire
    image_path = tmp_path / "test_image.jpg"
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Mock PIL Image
    with patch('src.multimodal.vision_processor.Image') as mock_image:
        mock_pil_image = Mock()
        mock_pil_image.convert.return_value = mock_pil_image
        mock_pil_image.size = (100, 100)
        mock_image.open.return_value.convert.return_value = mock_pil_image
        mock_image.fromarray.return_value.convert.return_value = mock_pil_image

        # Mock des méthodes
        vision_processor.generate_image_description = Mock(return_value=SceneDescription("Test description", confidence=0.8))
        vision_processor._detect_objects = Mock(return_value=ObjectDetection([{"label": "test", "confidence": 0.9}]))
        vision_processor._describe_scene = Mock(return_value=SceneDescription("Test scene", confidence=0.7))
        vision_processor._extract_dominant_colors = Mock(return_value=[("red", 0.5)])
        vision_processor.extract_text_from_image = Mock(return_value=OCRResult("Test text", confidence=0.8))

        result = vision_processor.analyze_image(image_path)

        assert isinstance(result, ImageAnalysis)
        assert result.description == "Test description"
        assert len(result.objects_detected) == 1
        assert result.scene_description == "Test scene"


def test_generate_image_description(vision_processor, sample_image):
    """Teste la génération de description d'image."""
    with patch.object(vision_processor, 'blip_model') as mock_model, \
         patch.object(vision_processor, 'blip_processor') as mock_processor:

        # Mock BLIP
        mock_processor.return_tensors = Mock(return_value={"input_ids": torch.tensor([[1, 2, 3]])})
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 0]])  # EOS token
        mock_processor.decode.return_value = "A beautiful test image"

        # Mock _prepare_image
        vision_processor._prepare_image = Mock(return_value=Mock())

        result = vision_processor.generate_image_description(sample_image, DetailLevel.MEDIUM)

        assert isinstance(result, SceneDescription)
        assert "beautiful test image" in result.description.lower()


def test_answer_visual_question(vision_processor, sample_image):
    """Teste les questions visuelles."""
    with patch.object(vision_processor, 'clip_model') as mock_model, \
         patch.object(vision_processor, 'clip_processor') as mock_processor:

        # Mock CLIP
        mock_processor.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        mock_model.return_value.logits_per_image = torch.tensor([[0.8]])

        vision_processor._prepare_image = Mock(return_value=Mock())
        vision_processor._detect_objects = Mock(return_value=ObjectDetection([]))

        result = vision_processor.answer_visual_question(
            sample_image, "What do you see?"
        )

        assert isinstance(result, VisualQuestion)
        assert result.question == "What do you see?"
        assert result.confidence > 0


def test_extract_text_from_image(vision_processor, tmp_path):
    """Teste l'extraction de texte."""
    image_path = tmp_path / "test.jpg"

    with patch.object(vision_processor, 'ocr_reader') as mock_reader:
        # Mock easyocr
        mock_reader.readtext.return_value = [
            ([[10, 10], [50, 10], [50, 30], [10, 30]], "Hello", 0.9),
            ([[60, 10], [100, 10], [100, 30], [60, 30]], "World", 0.8)
        ]

        vision_processor._prepare_image = Mock(return_value=Mock())

        result = vision_processor.extract_text_from_image(image_path)

        assert isinstance(result, OCRResult)
        assert "Hello World" in result.text
        assert result.confidence > 0
        assert len(result.bounding_boxes) == 2


def test_detect_objects(vision_processor, sample_image):
    """Teste la détection d'objets."""
    with patch.object(vision_processor, 'detr_model') as mock_model, \
         patch.object(vision_processor, 'detr_processor') as mock_processor:

        # Mock DETR
        mock_outputs = {
            "logits": torch.randn(1, 100, 92),
            "pred_boxes": torch.randn(1, 100, 4)
        }
        mock_model.return_value = mock_outputs

        mock_processor.post_process_object_detection.return_value = [{
            "scores": torch.tensor([0.9, 0.8]),
            "labels": torch.tensor([1, 2]),
            "boxes": torch.tensor([[10, 10, 50, 50], [60, 60, 80, 80]])
        }]

        mock_model.config.id2label = {1: "person", 2: "car"}

        vision_processor._prepare_image = Mock(return_value=Mock())

        result = vision_processor._detect_objects(sample_image)

        assert isinstance(result, ObjectDetection)
        assert len(result.objects) == 2
        assert result.objects[0]["label"] == "person"


def test_extract_dominant_colors(vision_processor, sample_image):
    """Teste l'extraction des couleurs dominantes."""
    vision_processor._prepare_image = Mock(return_value=Mock())

    # Mock PIL resize et array conversion
    mock_small_image = Mock()
    mock_small_image.resize.return_value = mock_small_image
    vision_processor._prepare_image.return_value.resize.return_value = mock_small_image

    with patch('numpy.array') as mock_np_array:
        mock_np_array.return_value = np.random.randint(0, 255, (50, 50, 3))

        result = vision_processor._extract_dominant_colors(sample_image)

        assert isinstance(result, list)
        assert len(result) == 5  # 5 clusters
        assert all(isinstance(color, tuple) and len(color) == 2 for color in result)


def test_get_image_embedding(vision_processor, sample_image):
    """Teste l'extraction d'embeddings d'image."""
    with patch.object(vision_processor, 'clip_model') as mock_model, \
         patch.object(vision_processor, 'clip_processor') as mock_processor:

        # Mock CLIP
        mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
        mock_model.get_image_features.return_value = torch.randn(1, 512)

        vision_processor._prepare_image = Mock(return_value=Mock())

        result = vision_processor.get_image_embedding(sample_image)

        assert isinstance(result, np.ndarray)
        assert result.shape == (512,)


def test_error_handling(vision_processor):
    """Teste la gestion d'erreurs."""
    # Image inexistante
    with pytest.raises(FileNotFoundError):
        vision_processor.analyze_image("nonexistent.jpg")

    # Modèle non disponible
    vision_processor.blip_model = None
    result = vision_processor.generate_image_description("dummy")
    assert "non disponible" in result.description