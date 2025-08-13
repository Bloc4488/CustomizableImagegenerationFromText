import json
import re
import shutil
from tkinter import Tk, filedialog

import cv2
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
from Model_with_text import StableDiffusionWithText
from Kandinsky_with_text import KandinskyWithText


def _choose_interpolation_type() -> str:
    valid_types = ["linear", "nonlinear", "spherical"]
    while True:
        try:
            type = input(f"Choose interpolation type({valid_types}): ")
            if type in valid_types:
                return type
        except ValueError:
            raise ValueError("Please enter a valid interpolation type.")


def _display_menu() -> None:
    menu = [
        "******************************",
        "* 1. Choose model            *",
        "* 2. Write prompt            *",
        "* 3. Choose style(s)         *",
        "* 4. Make image              *",
        "* 5. Save image              *",
        "* 6. Start full test system  *",
        "* 7. Load config from JSON   *",
        "* 9. Exit program            *",
        "******************************"
    ]
    print("\n".join(menu))


def _get_user_choice() -> int:
    while True:
        try:
            return int(input("Enter your choice: "))
        except ValueError:
            print("Please enter a valid number.")


class ImageGeneratorMenu:
    GENERATED_IMAGES_DIR = Path(__file__).parent / 'Generated Images'

    def __init__(self):
        self.model = None
        self.image = None
        self.prompt: Optional[str] = None
        self.styles: List[str] = [] #Raw style inputs
        self.style_prompts: List[str] = [] #Text style prompts
        self.style_folders: List[str] = [] #Folder style paths
        self.weights: List[float] = []
        self.interpolation_type: str

    def run(self) -> None:
        while True:
            _display_menu()
            choice = _get_user_choice()
            if not self._handle_choice(choice):
                break

    def _handle_choice(self, choice: int) -> bool:
        actions = {
            1: self._choose_model,
            2: self._write_prompt,
            3: self._choose_styles,
            4: self._generate_image,
            5: self._save_image,
            6: self._test_system,
            7: self._run_tests_from_json,
            9: lambda: False
        }
        action = actions.get(choice)
        if action:
            return action()
        print("Invalid choice.")
        return True


    def _choose_model(self, test_choice: int = 0) -> bool:
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
                if test_choice == 0:
                    choice = int(input("Choose model: "))
                else:
                    choice = test_choice
                if 1 <= choice <= len(models):
                    name, model_class, kwargs = models[choice - 1]
                    self.model = model_class(**kwargs)
                    print(f"Chosen model: {self.model.name}")
                    return True
                print("Invalid choice.")
            except ValueError:
                print("Please enter a valid number.")

    def _write_prompt(self, test_choice: int = 0) -> bool:
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
        if test_choice == 0:
            choice = input("Enter your prompt number or type your own: ").strip()
        else:
            choice = test_choice
        if choice.isdigit() and 1 <= int(choice) <= len(default_prompts):
            self.prompt = default_prompts[int(choice) - 1]
        else:
            self.prompt = choice
        print(f"Selected Prompt: {self.prompt}")
        return True

    def _choose_styles(self) -> bool:
        self.styles.clear()
        self.style_prompts.clear()
        self.style_folders.clear()
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
            style = input(f"Enter style {i + 1} (text or folder path): ").strip()
            while True:
                try:
                    weight = float(input(f"Enter weight {i + 1} (0.0 - 1.0): "))
                    if 0 <= weight <= 1:
                        break
                    print("Weight must be between 0.0 and 1.0.")
                except ValueError:
                    print("Please enter a valid number.")

            style_path = Path(style)
            if style_path.exists() and style_path.is_dir():
                self.style_folders.append(str(style_path))
                print(f"Style {i + 1}: '{style}' recognized as a folder.")
            else:
                self.style_prompts.append(style)
                print(f"Style {i + 1}: '{style}' recognized as a text prompt.")
            self.styles.append(style)
            self.weights.append(weight)
        return True

    def _test_system(self) -> bool:
        user_choice_model_test = input("Enter your choice model separated by space or 'all': ")
        if user_choice_model_test == "all":
            user_choice_model_test = [1, 2, 3, 4, 5]
        else:
            try:
                user_choice_model_test = list(map(int, user_choice_model_test.split()))
            except ValueError:
                print("Please enter a valid numbers.")

        user_choice_prompt_test = input("Enter your choice prompt number or 'all': ")
        if user_choice_prompt_test == "all":
            user_choice_prompt_test = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        else:
            try:
                user_choice_prompt_test = list(map(str, user_choice_prompt_test.split()))
            except ValueError:
                print("Please enter a valid numbers.")

        self._choose_styles()
        valid_interpolation_types = ["linear", "nonlinear", "spherical"]

        for model_choice in user_choice_model_test:
            self._choose_model(test_choice=model_choice)
            for prompt_choice in user_choice_prompt_test:
                self._write_prompt(test_choice=prompt_choice)
                for interpolation_type in valid_interpolation_types:
                    self.interpolation_type = interpolation_type
                    try:
                        self.image = self.model.generate_image(
                            prompt=self.prompt,
                            style_prompts=self.style_prompts,
                            style_folders=self.style_folders,
                            style_weights=self.weights,
                            interpolation_type=self.interpolation_type
                        )
                        self._save_image()
                    except Exception as e:
                        print("Error generating image:", e)
        return True

    def _run_tests_from_json(self) -> bool:
        Tk().withdraw()
        json_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not json_path:
            print("No JSON file selected.")
            return False

        with open(json_path, "r", encoding="utf-8") as f:
            config_list = json.load(f)

        if not isinstance(config_list, list):
            print("JSON must contain a list of configurations.")
            return False

        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        existing = [d for d in results_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        run_id = max([int(d.name) for d in existing], default=0) + 1
        run_path = results_dir / str(run_id)
        run_path.mkdir()
        shutil.copy(json_path, run_path / "config.json")

        # Map model IDs
        model_map = {
            "sd14": (StableDiffusionWithText, {"use_lora": False, "model_id": 1.4}),
            "sd15": (StableDiffusionWithText, {"use_lora": False, "model_id": 1.5}),
            "sd14_lora": (StableDiffusionWithText, {"use_lora": True, "model_id": 1.4}),
            "sd15_lora": (StableDiffusionWithText, {"use_lora": True, "model_id": 1.5}),
            "kandinsky": (KandinskyWithText, {})
        }

        for idx, config in enumerate(config_list):
            test_group = config.get("test_group", "Unknown")
            model_id = config.get("model", "sd15")
            if model_id not in model_map:
                print(f"[{idx}] Unknown model: {model_id}")
                continue

            model_class, kwargs = model_map[model_id]
            self.model = model_class(**kwargs)

            self.prompt = config.get("prompt", "")
            self.style_prompts = config.get("styles", [])
            self.style_weights = config.get("weights", [])
            self.interpolation_type = config.get("interpolation", "linear")
            swirl = config.get("swirl_factor")
            smoothness = config.get("smoothness")
            seed = config.get("seed")

            print(f"[{idx + 1}/{len(config_list)}] Generating for group {test_group}...")
            try:
                self.image = self.model.generate_image(
                    prompt=self.prompt,
                    style_prompts=self.style_prompts,
                    style_weights=self.style_weights,
                    interpolation_type=self.interpolation_type,
                    swirl_factor=swirl,
                    smoothness=smoothness,
                    seed=seed
                )
            except Exception as e:
                print(f"Generation failed: {e}")
                continue

            group_path = run_path / test_group
            group_path.mkdir(parents=True, exist_ok=True)

            prompt_part = re.sub(r'[<>:"/\\|?*]', '', self.prompt[:30]).replace(" ", "_")
            styles_part = ("_".join(f"{s}{w:.2f}" for s, w in zip(self.style_prompts, self.style_weights))
                           .replace(" ", "_").replace("/", "_"))
            model_part = self.model.name.replace(" ", "").replace(".", "")
            method_part = self.interpolation_type
            params_part = ""
            if swirl is not None:
                params_part += f"_swirl{swirl:.2f}"
            if smoothness is not None:
                params_part += f"_smooth{smoothness:.2f}"
            if seed is not None:
                params_part += f"_seed{seed}"

            filename = f"{prompt_part}_{styles_part}_{model_part}_{method_part}{params_part}.png"
            if len(filename) > 250:
                filename = filename[:250] + ".png"

            image_np = self._convert_to_numpy()
            output_path = group_path / filename
            if cv2.imwrite(str(output_path), image_np):
                print(f"Saved: {output_path}")
            else:
                print(f"Failed to save image.")

        print(f"\nAll images saved under: {run_path}")
        return True

    def _generate_image(self) -> bool:
        if not self.model or not self.prompt:
            print("Please choose a model and a prompt first.")
            return True
        self.interpolation_type = _choose_interpolation_type()

        try:
            self.image = self.model.generate_image(
                prompt=self.prompt,
                style_prompts=self.style_prompts,
                style_folders=self.style_folders,
                style_weights=self.weights,
                interpolation_type=self.interpolation_type
            )
            self._show_image()
        except Exception as e:
            print(f"Error generating image: {str(e)}")
        return True

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

        styles_str = "_".join(f"{s}{w:.2f}" for s, w in zip(self.styles, self.weights)).replace("/", "_")
        prompt_clean = re.sub(r'[<>:"/\\|?*]]', '', self.prompt[:30])
        image_name = f"{prompt_clean}_{styles_str}_{self.interpolation_type}".replace(" ", "_").replace(".", "_")
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
