import os

import cv2
import numpy as np
from LoRa_with_text import LoRa_with_text

GENERATED_IMAGES = "Generated Images"

class Menu():
    def __init__(self):
        self.model = None
        self.image = None
        self.prompt = None
        self.styles = []
        self.weights = []

    def choose_menu(self):
        choice = None
        while choice != 9 or not choice:
            self.print()
            choice = int(input("Enter your choice: "))
            if choice == 1:
                self.choose_model()
            elif choice == 2:
                self.write_prompt()
            elif choice == 3:
                self.choose_style()
            elif choice == 4:
                self.make_image()
            elif choice == 5:
                self.save_image()
            else:
                print("Invalid choice")
        print("End of program")

    def print(self):
        print("******************************")
        print("* 1. Choose model            *")
        print("* 2. Write prompt            *")
        print("* 3. Choose style(s)         *")
        print("* 4. Make image              *")
        print("* 5. Save image              *")
        print("* 9. Exit program            *")
        print("******************************")

    def choose_model(self):
        print("1. LoRa")
        rightChoice = False
        while(rightChoice == False):
            choice = int(input("Choose model: "))
            if choice == 1:
                self.model = LoRa_with_text()
                rightChoice = True
            else:
                print("Please choose a model")
                rightChoice = False
        print(f"Chosen model: {self.model}")

    def write_prompt(self):
        prompts = [
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
        for i, prompt in enumerate(prompts, 1):
            print(f"{i}. {prompt}")
        choice = input("Enter your prompt number or type your own: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(prompts):
            self.prompt = prompts[int(choice) - 1]
        else:
            self.prompt = choice
        print(f"Selected Prompt: {self.prompt}")

    def choose_style(self):
        self.styles = []
        self.weights = []
        style_count = int(input("Number of styles: "))
        for i in range(style_count):
            self.styles.append(input(f"Enter style {i + 1} (text or folder name):"))
            self.weights.append(float(input(f"Weight for style {i + 1} (0.0 - 1.0):")))

    def make_image(self):
        self.image = self.model.generate_image(prompt=self.prompt, style_prompts=self.styles, style_weights=self.weights)
        self.show_image()

    def show_image(self):
        image_np = self.convert_image_to_np()
        cv2.imshow(f"Generated Image with {self.styles}", image_np)
        cv2.waitKey(0)

    def save_image(self):
        styles_str = "_".join([f"{style}{weight:.2f}" for style, weight in zip(self.styles, self.weights)])
        image_name = f"{self.prompt[:30]}_{styles_str}.png".replace(" ", "_")
        image_path = os.path.join(GENERATED_IMAGES, image_name)
        image_np = self.convert_image_to_np()
        cv2.imwrite(image_path, image_np)
        print("Image saved at", image_path)

    def convert_image_to_np(self):
        image_np = np.array(self.image)
        if image_np.dtype in [np.float32, np.float16]:
            image_np = (image_np * 255).astype(np.uint8)
        image_np = image_np[..., ::-1]
        return image_np



