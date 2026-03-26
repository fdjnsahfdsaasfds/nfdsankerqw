# predict.py
import os
import shutil
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

        # Cast VAE to float32 BEFORE CPU offload so the offload hooks see the right dtype
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        self.pipe.vae.to(dtype=torch.float32)

        # REQUIRED FOR 14B MoE MODEL: offloads weights to CPU RAM between uses
        # to prevent OOM on Replicate GPUs
        self.pipe.enable_model_cpu_offload()

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
        lora_scale: float = Input(description="Scale/strength for the LoRA (0.0 to 2.0).", default=1.0, ge=0.0, le=2.0),
        num_frames: int = Input(description="Number of frames. Must satisfy (n-1) % 4 == 0, e.g. 17, 33, 49, 65, 81.", default=81, ge=5, le=121),
        num_inference_steps: int = Input(description="Number of denoising steps.", default=50, ge=1, le=100),
        guidance_scale: float = Input(description="Guidance scale.", default=5.0, ge=1.0, le=20.0),
        seed: int = Input(description="Random seed. Set to -1 to randomize.", default=-1),
    ) -> Path:
        """Run a single prediction on the model"""

        # Validate num_frames satisfies (n-1) % 4 == 0
        if (num_frames - 1) % 4 != 0:
            raise ValueError(f"num_frames must satisfy (n-1) % 4 == 0 (e.g. 17, 33, 49, 65, 81). Got {num_frames}.")

        # 0. Process the input image
        input_image = Image.open(str(image)).convert("RGB")

        temp_dir = None
        lora_path = None
        adapter_name = "custom_lora"

        # 1. Download and load LoRA dynamically if a URL is provided
        if lora_url:
            download_url = lora_url
            if civitai_token:
                delimiter = "&" if "?" in download_url else "?"
                download_url += f"{delimiter}token={civitai_token}"

            print("Downloading LoRA...")  # don't log URL — may contain token
            try:
                temp_dir = tempfile.mkdtemp()
                lora_path = os.path.join(temp_dir, "lora.safetensors")

                response = requests.get(download_url, stream=True, allow_redirects=True, timeout=300)
                response.raise_for_status()

                with open(lora_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print("Loading LoRA weights into pipeline...")
                self.pipe.load_lora_weights(lora_path, adapter_name=adapter_name)
                self.pipe.set_adapters([adapter_name], adapter_weights=[lora_scale])
            except Exception as e:
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                raise RuntimeError(f"Failed to load LoRA: {e}") from e
        else:
            self.pipe.unload_lora_weights()

        # 2. Setup seed
        if seed == -1:
            seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
        generator = torch.Generator(device="cuda").manual_seed(seed)

        # 3. Generate video
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

        # 4. Cleanup LoRA and temp files
        if lora_url:
            self.pipe.unload_lora_weights()
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        # 5. Export to video and return
        out_path = "/tmp/output.mp4"
        export_to_video(output, out_path, fps=16)

        return Path(out_path)
