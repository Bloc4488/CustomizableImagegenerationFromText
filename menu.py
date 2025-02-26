import re
import cv2
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
from Model_with_text import StableDiffusionWithText
from Kandinsky_with_text import KandinskyWithText

class ImageGeneratorMenu:
    GENERATED_IMAGES_DIR = Path(__file__).parent / 'Generated Images'

    def __init__(self):
        self.model = None
        self.image = None
        self.prompt: Optional[str] = None
        self.styles: List[str] = []
        self.weights: List[float] = []

    def run(self) -> None:
        while True:
            self._display_menu()
            choice = self._get_user_choice()
            if not self._handle_choice(choice):
                break

    def _display_menu(self) -> None:
        menu = [
            "******************************",
            "* 1. Choose model            *",
            "* 2. Write prompt            *",
            "* 3. Choose style(s)         *",
            "* 4. Make image              *",
            "* 5. Save image              *",
            "* 9. Exit program            *",
            "******************************"
        ]
        print("\n".join(menu))

    def _get_user_choice(self) -> int:
        while True:
            try:
                return int(input("Enter your choice: "))
            except ValueError:
                print("Please enter a valid number.")

    def _handle_choice(self, choice: int) -> bool:
        actions = {
            1: self._choose_model,
            2: self._write_prompt,
            3: self._choose_styles,
            4: self._generate_image,
            5: self._save_image,
            9: lambda: False
        }
        action = actions.get(choice)
        if action:
            return action()
        print("Invalid choice.")
        return True


    def _choose_model(self) -> bool:
        models = [
            ("Stable Diffusion 1.5", StableDiffusionWithText, {"use_lora": False, "model_id": 1.5}),
            ("Stable Diffusion 1.4", StableDiffusionWithText, {"use_lora": False, "model_id": 1.4}),
            ("Stable Diffusion 1.5 with LoRA", StableDiffusionWithText, {"use_lora": True, "model_id": 1.5}),
            ("Stable Diffusion 1.4 with LoRA", StableDiffusionWithText, {"use_lora": True, "model_id": 1.4}),
            ("Kandinsky 2.2", KandinskyWithText, {})
        ]

        for i, (name, _, _) in enumerate(models, 1):
            print(f"{i}. {name}")

        while True:
            try:
                choice = int(input("Choose model: "))
                if 1 <= choice <= len(models):
                    name, model_class, kwargs = models[choice - 1]
                    self.model = model_class(**kwargs)
                    print(f"Chosen model: {self.model.name}")
                    return True
                print("Invalid choice.")
            except ValueError:
                print("Please enter a valid number.")

    def _write_prompt(self) -> bool:
        default_prompts = [
            "A serene landscape with a peaceful lake reflecting the sunset, surrounded by mountains and lush trees.",
            "A quiet village covered in fresh snow, with warm lights glowing from the windows of wooden cottages.",
            "A dense jungle with towering ancient trees, vines hanging down, and a hidden temple partially covered by moss.",
            "A futuristic city filled with towering skyscrapers, flying cars, and bustling streets illuminated by artificial lights.",
            "A mysterious figure standing on the edge of a cliff, looking at a vast ocean with storm clouds rolling in.",
            "An astronaut floating in deep space, surrounded by distant galaxies and a glowing nebula.",
            "A vast desert with endless dunes, an abandoned caravan partially buried in the sand, and a lone traveler walking under the scorching sun.",
            "An enchanted forest with glowing mushrooms, floating wisps of light, and an ancient stone pathway leading deeper into the woods.",
            "A medieval battlefield at dawn, with banners flying, armored knights preparing for battle, and a castle in the distance.",
            "A bustling marketplace in an ancient city, filled with merchants selling exotic goods, colorful fabrics, and the sound of people haggling."
        ]
        for i, prompt in enumerate(default_prompts, 1):
            print(f"{i}. {prompt}")
        choice = input("Enter your prompt number or type your own: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(default_prompts):
            self.prompt = default_prompts[int(choice) - 1]
        else:
            self.prompt = choice
        print(f"Selected Prompt: {self.prompt}")
        return True

    def _choose_styles(self) -> bool:
        self.styles.clear()
        self.weights.clear()
        while True:
            try:
                style_count = int(input("Enter number of styles: "))
                if style_count >= 0:
                    break
                print("Please enter a non-negative number.")
            except ValueError:
                print("Please enter a valid number.")

        for i in range(style_count):
            style = input(f"Enter style {i + 1}: ")
            while True:
                try:
                    weight = float(input(f"Enter weight {i + 1} (0.0 - 1.0): "))
                    if 0 <= weight <= 1:
                        break
                    print("Weight must be between 0.0 and 1.0.")
                except ValueError:
                    print("Please enter a valid number.")
            self.styles.append(style)
            self.weights.append(weight)
        return True

    def _generate_image(self) -> bool:
        if not self.model or not self.prompt:
            print("Please choose a model and a prompt first.")
            return True
        interpolation_type = self._choose_interpolation_type()

        self.image = self.model.generate_image(
            prompt=self.prompt,
            style_prompts=self.styles,
            style_weights=self.weights,
            interpolation_type=interpolation_type
        )
        self._show_image()
        return True

    def _choose_interpolation_type(self) -> str:
        valid_types = ["linear", "nonlinear"]
        while True:
            try:
                type = input(f"Choose interpolation type({valid_types}): ")
                if type in valid_types:
                    return type
            except ValueError:
                raise ValueError("Please enter a valid interpolation type.")


    def _show_image(self) -> None:
        if self.image is not None:
            image_np = self._convert_to_numpy()
            styles_str = ", ".join(self.styles)
            cv2.imshow(f"Generated Image with {styles_str}", image_np)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def _save_image(self) -> bool:
        if self.image is None:
            print("No image to save. Generate an image first.")
            return True

        styles_str = "_".join(f"{s}{w:.2f}" for s, w in zip(self.styles, self.weights))
        prompt_clean = re.sub(r'[<>:"/\\|?*]]', '', self.prompt[:30])
        image_name = f"{prompt_clean}_{styles_str}".replace(" ", "_").replace(".", "_")
        if len(image_name) > 250:
            image_name = image_name[:250]
        image_name += ".png"

        save_dir = self.GENERATED_IMAGES_DIR / self.model.name
        save_dir.mkdir(parents=True, exist_ok=True)
        image_path = save_dir / image_name

        image_np = self._convert_to_numpy()
        if cv2.imwrite(str(image_path), image_np):
            print(f"Image saved at {image_path}")
        else:
            print("Image could not be saved.")
        return True

    def _convert_to_numpy(self) -> np.ndarray:
        image_np = np.array(self.image)
        if image_np.dtype in (np.float32, np.float16):
            image_np = (image_np * 255).astype(np.uint8)
        return image_np[..., ::-1]
