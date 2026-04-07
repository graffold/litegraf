"""
Vision Processor for Biomedical Images
Uses Llama-3.2-11B-Vision-Instruct via AWS Bedrock to analyze protein diagrams,
pathway charts, and other biomedical images.
"""

import base64
import json
from typing import Any

from src.config import Config
from pipeline.processors.multimodal_processor import (MultimodalDocument,
                                                 MultimodalProcessor)
from src.utils.logging_utils import setup_logging

logger = setup_logging()


class VisionProcessor:
    """
    Processes biomedical images using vision-language models.
    Extracts entities (proteins, pathways, diseases) from diagrams and charts.
    """

    def __init__(
        self,
        vision_model_id: str | None = None,
        region_name: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        """
        Initialize vision processor.

        Args:
            vision_model_id: Bedrock model ID for vision tasks (defaults to VISION_MODEL_ID config)
            region_name: AWS region (defaults to VISION_MODEL_REGION config)
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
        """
        self.vision_model_id = (
            vision_model_id
            or Config.get_config("VISION_MODEL_ID")
        )
        self.region_name = (
            region_name
            or Config.get_config("VISION_MODEL_REGION")
            or Config.get_config("AWS_REGION")
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize Bedrock vision client
        try:
            import boto3

            self.bedrock_runtime = boto3.client(
                service_name="bedrock-runtime", region_name=self.region_name
            )
            logger.info(
                f"Initialized VisionProcessor with {vision_model_id} "
                f"in region {self.region_name}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise

    def analyze_image(
        self, image_data: bytes, context: str | None = None, image_type: str = "figure"
    ) -> dict[str, Any]:
        """
        Analyze a biomedical image and extract structured information.

        Args:
            image_data: Raw image bytes
            context: Optional text context from surrounding document
            image_type: Type of image (figure, diagram, pathway, chart)

        Returns:
            Dict with extracted entities, caption, and description
        """
        # Resize image to optimize token usage
        resized_image = MultimodalProcessor.resize_image(
            image_data, max_size=(1024, 1024)
        )
        image_base64 = base64.b64encode(resized_image).decode("utf-8")

        # Create prompt based on image type
        prompt = self._create_analysis_prompt(image_type, context)

        # Call Bedrock with vision model
        try:
            response = self._invoke_vision_model(prompt, image_base64)
            parsed_result = self._parse_vision_response(response, image_type)

            logger.info(
                f"Analyzed {image_type} image, "
                f"extracted {len(parsed_result.get('entities', []))} entities"
            )

            return parsed_result

        except Exception as e:
            logger.error(f"Vision analysis failed: {e}", exc_info=True)
            return {"caption": "", "description": "", "entities": [], "error": str(e)}

    def _create_analysis_prompt(
        self, image_type: str, context: str | None = None
    ) -> str:
        """
        Create specialized prompt for biomedical image analysis.

        Args:
            image_type: Type of image
            context: Optional surrounding text

        Returns:
            Formatted prompt
        """
        base_prompt = """You are an expert in biomedical image analysis, specializing in proteomics and molecular biology.

Analyze this biomedical image and extract the following information in JSON format:

{
  "caption": "Brief descriptive caption for the image",
  "description": "Detailed description of what the image shows",
  "image_type": "pathway|protein_structure|protein_interaction|expression_chart|western_blot|microscopy|other",
  "entities": [
    {
      "name": "Entity name (protein, gene, disease, pathway, etc.)",
      "type": "Protein|Disease|Pathway|CellularComponent|BiologicalProcess|Gene",
      "confidence": 0.0-1.0,
      "context": "Brief context about this entity in the image"
    }
  ],
  "relationships": [
    {
      "source": "Entity 1 name",
      "target": "Entity 2 name",
      "type": "interacts_with|regulates|activates|inhibits|part_of",
      "confidence": 0.0-1.0
    }
  ],
  "key_findings": ["Finding 1", "Finding 2", ...]
}
"""

        if image_type == "pathway":
            base_prompt += """
Focus on:
- Signaling pathways and their components
- Protein-protein interactions
- Regulatory mechanisms (activation, inhibition)
- Cellular locations of pathway components
"""
        elif image_type == "protein_structure":
            base_prompt += """
Focus on:
- Protein domains and structural features
- Binding sites and active sites
- Post-translational modifications
- Structural motifs
"""
        elif image_type == "chart":
            base_prompt += """
Focus on:
- Proteins or genes being measured
- Experimental conditions
- Statistical significance
- Quantitative relationships
"""

        if context:
            base_prompt += f"\n\nSurrounding document context:\n{context[:500]}\n"

        base_prompt += (
            "\nProvide your analysis as valid JSON matching the schema above."
        )

        return base_prompt

    def _invoke_vision_model(self, prompt: str, image_base64: str) -> str:
        """
        Invoke Bedrock vision model with image and text.

        Args:
            prompt: Text prompt
            image_base64: Base64-encoded image

        Returns:
            Model response text
        """
        # Llama 3.2 Vision format for Bedrock
        request_body = {
            "prompt": prompt,
            "images": [image_base64],
            "max_gen_len": self.max_tokens,
            "temperature": self.temperature,
            "top_p": 0.9,
        }

        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=self.vision_model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )

            response_body = json.loads(response["body"].read())

            # Extract response text (format may vary by model version)
            if "generation" in response_body:
                return response_body["generation"]
            if "completions" in response_body:
                return response_body["completions"][0]["data"]["text"]
            if "outputs" in response_body:
                return response_body["outputs"][0]["text"]
            logger.warning(f"Unexpected response format: {response_body.keys()}")
            return str(response_body)

        except Exception as e:
            logger.error(f"Bedrock invocation failed: {e}", exc_info=True)
            raise

    def _parse_vision_response(
        self, response_text: str, image_type: str
    ) -> dict[str, Any]:
        """
        Parse vision model response into structured format.

        Args:
            response_text: Raw response from vision model
            image_type: Type of image analyzed

        Returns:
            Parsed structured data
        """
        try:
            # Try to extract JSON from response
            # Handle cases where model wraps JSON in markdown code blocks
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text

            parsed = json.loads(json_text)

            # Validate required fields
            return {
                "caption": parsed.get("caption", ""),
                "description": parsed.get("description", ""),
                "image_type": parsed.get("image_type", image_type),
                "entities": parsed.get("entities", []),
                "relationships": parsed.get("relationships", []),
                "key_findings": parsed.get("key_findings", []),
            }

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text[:500]}")

            # Fallback: return raw text as description
            return {
                "caption": "",
                "description": response_text,
                "image_type": image_type,
                "entities": [],
                "relationships": [],
                "key_findings": [],
            }

    def process_document_images(
        self, multimodal_doc: MultimodalDocument, max_images: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Process all images in a multimodal document.

        Args:
            multimodal_doc: Document containing images
            max_images: Optional limit on number of images to process

        Returns:
            List of analysis results
        """
        results = []
        images_to_process = (
            multimodal_doc.images[:max_images] if max_images else multimodal_doc.images
        )

        logger.info(
            f"Processing {len(images_to_process)} images from document {multimodal_doc.doc_id}"
        )

        for image in images_to_process:
            try:
                # Get context from surrounding text (simplified - improve this logic)
                context = multimodal_doc.text_content[:1000]

                analysis = self.analyze_image(
                    image_data=image.image_data,
                    context=context,
                    image_type=image.image_type,
                )

                results.append(
                    {
                        "image_id": image.image_id,
                        "page": image.page_number,
                        "analysis": analysis,
                    }
                )

            except Exception as e:
                logger.error(f"Failed to process image {image.image_id}: {e}")
                results.append(
                    {
                        "image_id": image.image_id,
                        "page": image.page_number,
                        "error": str(e),
                    }
                )

        logger.info(f"Completed analysis of {len(results)} images")
        return results
