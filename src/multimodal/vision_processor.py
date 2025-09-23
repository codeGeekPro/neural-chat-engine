"""Processeur de vision pour l'analyse d'images et Q&A visuel."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np

# Imports conditionnels pour les bibliothèques de vision
try:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import torchvision.transforms as transforms
    from transformers import (
        CLIPProcessor, CLIPModel,
        BlipProcessor, BlipForConditionalGeneration,
        DetrImageProcessor, DetrForObjectDetection,
        OwlViTProcessor, OwlViTForObjectDetection
    )
    VISION_LIBRARIES_AVAILABLE = True
except ImportError:
    VISION_LIBRARIES_AVAILABLE = False
    torch = None
    Image = None
    transforms = None
    CLIPProcessor = None
    CLIPModel = None
    BlipProcessor = None
    BlipForConditionalGeneration = None
    DetrImageProcessor = None
    DetrForObjectDetection = None
    OwlViTProcessor = None
    OwlViTForObjectDetection = None

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    easyocr = None

from .multimodal_types import (
    VisionModelType,
    DetailLevel,
    Language,
    ImageAnalysis,
    VisualQuestion,
    OCRResult,
    ObjectDetection,
    SceneDescription,
    ImageInput,
    ImageEmbedding
)


logger = logging.getLogger(__name__)


class VisionProcessor:
    """Processeur de vision pour l'analyse d'images."""

    def __init__(
        self,
        vision_model: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        max_image_size: Tuple[int, int] = (224, 224)
    ):
        """Initialise le processeur de vision.

        Args:
            vision_model: Modèle de vision à utiliser
            device: Périphérique de calcul
            cache_dir: Répertoire de cache pour les modèles
            max_image_size: Taille maximale des images
        """
        if not VISION_LIBRARIES_AVAILABLE:
            raise ImportError(
                "Bibliothèques de vision non disponibles. Installez transformers, "
                "torch, pillow, et torchvision."
            )

        self.vision_model_name = vision_model
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_image_size = max_image_size

        # Modèles
        self.clip_model = None
        self.clip_processor = None
        self.blip_model = None
        self.blip_processor = None
        self.detr_model = None
        self.detr_processor = None
        self.owl_vit_model = None
        self.owl_vit_processor = None

        # OCR
        self.ocr_reader = None

        # Transformations d'image
        self.transform = transforms.Compose([
            transforms.Resize(max_image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Initialisation des modèles
        self._initialize_models()

        logger.info(f"VisionProcessor initialisé avec {vision_model}")

    def _initialize_models(self) -> None:
        """Initialise les modèles de vision."""
        cache_dir = str(self.cache_dir) if self.cache_dir else None

        # CLIP pour l'analyse générale
        try:
            self.clip_model = CLIPModel.from_pretrained(
                self.vision_model_name,
                cache_dir=cache_dir
            ).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(
                self.vision_model_name,
                cache_dir=cache_dir
            )
            logger.info("Modèle CLIP chargé")
        except Exception as e:
            logger.warning(f"Impossible de charger CLIP: {e}")

        # BLIP pour la génération de descriptions
        try:
            blip_model_name = "Salesforce/blip-image-captioning-base"
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                blip_model_name,
                cache_dir=cache_dir
            ).to(self.device)
            self.blip_processor = BlipProcessor.from_pretrained(
                blip_model_name,
                cache_dir=cache_dir
            )
            logger.info("Modèle BLIP chargé")
        except Exception as e:
            logger.warning(f"Impossible de charger BLIP: {e}")

        # DETR pour la détection d'objets
        try:
            detr_model_name = "facebook/detr-resnet-50"
            self.detr_model = DetrForObjectDetection.from_pretrained(
                detr_model_name,
                cache_dir=cache_dir
            ).to(self.device)
            self.detr_processor = DetrImageProcessor.from_pretrained(
                detr_model_name,
                cache_dir=cache_dir
            )
            logger.info("Modèle DETR chargé")
        except Exception as e:
            logger.warning(f"Impossible de charger DETR: {e}")

        # OWL-ViT pour la détection d'objets avec texte
        try:
            owl_model_name = "google/owlvit-base-patch32"
            self.owl_vit_model = OwlViTForObjectDetection.from_pretrained(
                owl_model_name,
                cache_dir=cache_dir
            ).to(self.device)
            self.owl_vit_processor = OwlViTProcessor.from_pretrained(
                owl_model_name,
                cache_dir=cache_dir
            )
            logger.info("Modèle OWL-ViT chargé")
        except Exception as e:
            logger.warning(f"Impossible de charger OWL-ViT: {e}")

        # OCR
        if OCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(['en', 'fr'])
                logger.info("OCR initialisé")
            except Exception as e:
                logger.warning(f"Impossible d'initialiser OCR: {e}")

    def analyze_image(
        self,
        image_path: Union[str, Path],
        question: Optional[str] = None
    ) -> ImageAnalysis:
        """Analyse une image.

        Args:
            image_path: Chemin vers l'image
            question: Question optionnelle sur l'image

        Returns:
            Analyse complète de l'image
        """
        start_time = time.time()
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image non trouvée: {image_path}")

        # Charge l'image
        image = self._load_image(image_path)

        # Analyse complète
        description = self.generate_image_description(image, DetailLevel.MEDIUM)
        objects = self._detect_objects(image)
        scene_desc = self._describe_scene(image)
        colors = self._extract_dominant_colors(image)
        ocr_text = self.extract_text_from_image(image_path)

        processing_time = time.time() - start_time

        return ImageAnalysis(
            image_path=image_path,
            description=description.description,
            objects_detected=objects.objects,
            scene_description=scene_desc.description,
            dominant_colors=colors,
            text_extracted=ocr_text.text if ocr_text else "",
            confidence_scores={
                "description": description.confidence,
                "scene": scene_desc.confidence,
                "ocr": ocr_text.confidence if ocr_text else 0.0
            },
            processing_time=processing_time
        )

    def generate_image_description(
        self,
        image: ImageInput,
        detail_level: DetailLevel = DetailLevel.MEDIUM
    ) -> SceneDescription:
        """Génère une description d'image.

        Args:
            image: Image à décrire
            detail_level: Niveau de détail souhaité

        Returns:
            Description de l'image
        """
        start_time = time.time()

        if self.blip_model is None:
            return SceneDescription(
                description="Modèle BLIP non disponible",
                confidence=0.0,
                processing_time=time.time() - start_time
            )

        # Prépare l'image
        pil_image = self._prepare_image(image)

        # Génère la description selon le niveau de détail
        if detail_level == DetailLevel.BRIEF:
            prompt = "a photo of"
            max_length = 10
        elif detail_level == DetailLevel.MEDIUM:
            prompt = "a photograph of"
            max_length = 30
        else:  # DETAILED
            prompt = "Describe this image in detail:"
            max_length = 50

        inputs = self.blip_processor(pil_image, prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.blip_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

        description = self.blip_processor.decode(outputs[0], skip_special_tokens=True)

        processing_time = time.time() - start_time

        return SceneDescription(
            description=description,
            confidence=0.8,  # Estimation
            processing_time=processing_time
        )

    def answer_visual_question(
        self,
        image: ImageInput,
        question: str,
        context: Optional[str] = None
    ) -> VisualQuestion:
        """Répond à une question sur une image.

        Args:
            image: Image concernée
            question: Question à poser
            context: Contexte additionnel

        Returns:
            Réponse à la question visuelle
        """
        start_time = time.time()
        image_path = Path("temp_image.jpg")  # Pour les métadonnées

        if self.clip_model is None:
            return VisualQuestion(
                question=question,
                image_path=image_path,
                answer="Modèle CLIP non disponible pour les questions visuelles",
                confidence=0.0,
                processing_time=time.time() - start_time
            )

        # Prépare l'image et la question
        pil_image = self._prepare_image(image)
        inputs = self.clip_processor(
            text=[question],
            images=pil_image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        confidence = probs[0][0].item()

        # Pour l'instant, réponse simplifiée
        # En production, utiliser un modèle VQA spécialisé
        if "what" in question.lower() or "which" in question.lower():
            # Essaie d'extraire des objets
            objects = self._detect_objects(pil_image)
            if objects.objects:
                top_object = objects.get_top_objects(1)[0]
                answer = f"I see a {top_object['label']} in the image."
            else:
                answer = "I can see various elements in the image."
        elif "how many" in question.lower():
            objects = self._detect_objects(pil_image)
            answer = f"I can detect {len(objects.objects)} objects in the image."
        elif "color" in question.lower():
            colors = self._extract_dominant_colors(pil_image)
            if colors:
                answer = f"The dominant color appears to be {colors[0][0]}."
            else:
                answer = "I can see various colors in the image."
        else:
            answer = "This appears to be an image with various visual elements."

        processing_time = time.time() - start_time

        return VisualQuestion(
            question=question,
            image_path=image_path,
            answer=answer,
            confidence=confidence,
            reasoning="Based on visual analysis using CLIP model",
            processing_time=processing_time
        )

    def extract_text_from_image(self, image: ImageInput) -> Optional[OCRResult]:
        """Extrait le texte d'une image (OCR).

        Args:
            image: Image contenant du texte

        Returns:
            Texte extrait avec métadonnées
        """
        start_time = time.time()

        if self.ocr_reader is None:
            logger.warning("OCR non disponible")
            return None

        # Prépare l'image
        pil_image = self._prepare_image(image)

        # Convertit en array numpy pour easyocr
        image_array = np.array(pil_image)

        # Effectue l'OCR
        results = self.ocr_reader.readtext(image_array)

        # Combine les résultats
        extracted_text = " ".join([result[1] for result in results])
        confidence = np.mean([result[2] for result in results]) if results else 0.0

        # Bounding boxes
        bounding_boxes = [
            {
                "text": result[1],
                "bbox": result[0],
                "confidence": result[2]
            }
            for result in results
        ]

        processing_time = time.time() - start_time

        return OCRResult(
            text=extracted_text,
            confidence=confidence,
            bounding_boxes=bounding_boxes,
            processing_time=processing_time
        )

    def _detect_objects(self, image: ImageInput) -> ObjectDetection:
        """Détecte les objets dans une image."""
        start_time = time.time()

        if self.detr_model is None:
            return ObjectDetection(processing_time=time.time() - start_time)

        pil_image = self._prepare_image(image)

        inputs = self.detr_processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.detr_model(**inputs)

        # Convertit les outputs en prédictions
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
        results = self.detr_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.5
        )[0]

        objects = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            objects.append({
                "label": self.detr_model.config.id2label[label.item()],
                "confidence": score.item(),
                "bbox": box.tolist()
            })

        processing_time = time.time() - start_time

        return ObjectDetection(
            objects=objects,
            processing_time=processing_time
        )

    def _describe_scene(self, image: ImageInput) -> SceneDescription:
        """Décrit la scène d'une image."""
        # Utilise BLIP pour une description de scène
        description = self.generate_image_description(image, DetailLevel.MEDIUM)

        # Analyse basique pour extraire des tags
        desc_text = description.description.lower()
        tags = []

        if any(word in desc_text for word in ["outdoor", "outside", "nature", "sky"]):
            tags.append("outdoor")
        if any(word in desc_text for word in ["indoor", "inside", "room", "building"]):
            tags.append("indoor")
        if any(word in desc_text for word in ["person", "people", "man", "woman"]):
            tags.append("people")
        if any(word in desc_text for word in ["animal", "dog", "cat", "bird"]):
            tags.append("animals")

        return SceneDescription(
            description=description.description,
            tags=tags,
            confidence=description.confidence,
            processing_time=description.processing_time
        )

    def _extract_dominant_colors(self, image: ImageInput) -> List[Tuple[str, float]]:
        """Extrait les couleurs dominantes d'une image."""
        pil_image = self._prepare_image(image)

        # Redimensionne pour l'analyse
        small_image = pil_image.resize((50, 50))
        pixels = np.array(small_image)

        # Remodel pour clustering
        pixels = pixels.reshape(-1, 3)

        # Clustering simple pour trouver les couleurs dominantes
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)

        # Couleurs dominantes
        colors = []
        for center in kmeans.cluster_centers_:
            # Convertit RGB en nom de couleur approximatif
            r, g, b = center.astype(int)
            color_name = self._rgb_to_color_name(r, g, b)
            percentage = 1.0 / 5  # Approximation simple
            colors.append((color_name, percentage))

        return colors

    def _rgb_to_color_name(self, r: int, g: int, b: int) -> str:
        """Convertit RGB en nom de couleur approximatif."""
        # Mapping simple des couleurs
        if r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif r > g + 50 and r > b + 50:
            return "red"
        elif g > r + 50 and g > b + 50:
            return "green"
        elif b > r + 50 and b > g + 50:
            return "blue"
        elif r > 150 and g > 150:
            return "yellow"
        elif r > 150 and b > 150:
            return "magenta"
        elif g > 150 and b > 150:
            return "cyan"
        else:
            return "gray"

    def _load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Charge une image depuis un fichier."""
        if Image is None:
            raise ImportError("PIL non disponible")

        return Image.open(image_path).convert('RGB')

    def _prepare_image(self, image: ImageInput) -> Image.Image:
        """Prépare une image pour le traitement."""
        if isinstance(image, (str, Path)):
            return self._load_image(image)
        elif hasattr(image, 'convert'):  # PIL Image
            return image.convert('RGB')
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert('RGB')
        else:
            raise ValueError(f"Type d'image non supporté: {type(image)}")

    def get_image_embedding(self, image: ImageInput) -> ImageEmbedding:
        """Extrait l'embedding d'une image."""
        if self.clip_model is None:
            raise RuntimeError("Modèle CLIP non disponible")

        pil_image = self._prepare_image(image)
        inputs = self.clip_processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)

        return image_features.cpu().numpy().flatten()