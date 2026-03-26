# predict.py
import os
import tempfile
import requests
from PIL import Image
from cog import BasePredictor, Input, Path
import torch
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video

MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model from weights baked into the image"""
        print(f"Loading {MODEL_ID} from cache...")
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

        # REQUIRED FOR 14B MoE MODEL: Offloads weights to CPU RAM when not actively in use
        # This prevents Out-Of-Memory (OOM) errors on Replicate's GPUs.
        self.pipe.enable_model_cpu_offload()

        # Diffusers recommends casting VAE to float32 for better decoding quality
        self.pipe.vae.to(dtype=torch.float32)

    def predict(
        self,
        image: Path = Input(description="Input image to animate."),
        prompt: str = Input(description="Prompt for video generation (tell the model how to animate the image)."),
        negative_prompt: str = Input(description="Negative prompt.", default=""),
        lora_url: str = Input(
            description="Optional: URL to a .safetensors LoRA file (e.g., from CivitAI).",
            default=None
        ),
        civitai_token: str = Input(
            description="Optional: CivitAI API token. Necessary if the LoRA requires an account/login to download.",
            default=None
        ),
        lora_scale: float = Input(description="Scale/strength for the LoRA.", default=1.0),
        num_frames: int = Input(description="Number of frames. (Should be 4*k + 1, e.g., 81).", default=81),
        num_inference_steps: int = Input(description="Number of denoising steps.", default=50),
        guidance_scale: float = Input(description="Guidance scale.", default=5.0),
        seed: int = Input(description="Random seed. Set to -1 to randomize.", default=-1),
    ) -> Path:
        """Run a single prediction on the model"""

        # 0. Process the input image
        input_image = Image.open(str(image)).convert("RGB")

        lora_path = None
        adapter_name = "custom_lora"

        # 1. Download and load LoRA dynamically if a URL is provided
        if lora_url:
            print(f"Downloading LoRA from {lora_url}...")
            temp_dir = tempfile.mkdtemp()
            lora_path = os.path.join(temp_dir, "lora.safetensors")

            if civitai_token:
                delimiter = "&" if "?" in lora_url else "?"
                lora_url += f"{delimiter}token={civitai_token}"

            response = requests.get(lora_url, stream=True, allow_redirects=True)
            response.raise_for_status()

            with open(lora_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print("Loading LoRA weights into pipeline...")
            self.pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
            self.pipe.set_adapters([adapter_name], adapter_weights=[lora_scale])
        else:
            self.pipe.unload_lora_weights()

        # 2. Setup Seed
        if seed == -1:
            seed = int(torch.randint(0, 1000000, (1,)).item())
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # 3. Generate Video
        print(f"Generating video with seed {seed}...")
        kwargs = {
            "image": input_image,
            "prompt": prompt,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        output = self.pipe(**kwargs).frames[0]

        # 4. Cleanup LoRA to free up VRAM for the next request
        if lora_url:
            self.pipe.unload_lora_weights()
            if os.path.exists(lora_path):
                os.remove(lora_path)

        # 5. Export to video and return
        out_path = "/tmp/output.mp4"
        export_to_video(output, out_path, fps=16)

        return Path(out_path)
