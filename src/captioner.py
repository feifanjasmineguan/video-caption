import torch
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class Captioner:
    def __init__(self, device=None):
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading BLIP on {self.device}...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        self.model.eval()
        print("Captioner ready ✅")

    def caption_frame(self, frame_path: str) -> str:
        img_bgr = cv2.imread(frame_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image   = Image.fromarray(img_rgb)
        return self._generate(image)

    def _generate(self, image, repetition_penalty=3.0):
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=50,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=3,
                num_beams=5,
                early_stopping=True,
            )
        caption = self.processor.decode(output[0], skip_special_tokens=True)

        # Detect loop — retry with stronger penalty
        words = caption.split()
        if any(words.count(w) > 3 for w in words):
            return self._generate(image, repetition_penalty=5.0)

        return caption