import base64

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import torch
import random
import einops
import numpy as np
import cv2
import config
import os
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the model
save_path = "C:\\Users\Daria\Downloads\saved_model1.pth"
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(save_path, location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)
apply_uniformer = UniformerDetector()


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps,
            guess_mode, strength, scale, seed, eta):
    results = []
    with torch.no_grad():
        # Preprocess the input image
        input_image = HWC3(input_image)
        detected_map = apply_uniformer(resize_image(input_image, detect_resolution))
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        # Prepare control tensor
        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        # Set the random seed if needed
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        # Adjust memory settings if configured
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        # Prepare conditioning and unconditioning
        cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]
        }
        un_cond = {
            "c_concat": None if guess_mode else [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        # Apply control scales
        model.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else [strength] * 13)

        # Run sampling
        samples, intermediates = ddim_sampler.sample(
            ddim_steps, num_samples, shape, cond, verbose=False, eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        # Decode the generated samples
        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0,
                                                                                                           255).astype(
            np.uint8)

        # Take the first generated result from the batch as the single output image
        results.append(x_samples[0])  # Only taking the first image in each batch

    # Return the processed results for each input image
    return results


@app.route('/process', methods=['POST'])
def process_request():
    # Parse input data
    data = request.json
    image_data = data['image']  # Base64 string
    prompt = data['prompt']
    a_prompt = data.get('a_prompt', '')
    n_prompt = data.get('n_prompt', '')
    num_samples = data.get('num_samples', 1)
    image_resolution = data.get('image_resolution', 256)
    detect_resolution = data.get('detect_resolution', 256)
    ddim_steps = data.get('ddim_steps', 50)
    guess_mode = data.get('guess_mode', False)
    strength = data.get('strength', 1.0)
    scale = data.get('scale', 7.5)
    seed = data.get('seed', -1)
    eta = data.get('eta', 0.0)

    # Convert base64 image to OpenCV format
    image_data = np.frombuffer(io.BytesIO(base64.b64decode(image_data)).read(), np.uint8)
    input_image = cv2.imdecode(image_data, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('curent.jpg',input_image)

    # Process image
    output_image = process(
        input_image,
        prompt,
        a_prompt,
        n_prompt,
        num_samples,
        image_resolution,
        detect_resolution,
        ddim_steps,
        guess_mode,
        strength,
        scale,
        seed,
        eta
    )[0]

    # Save output as a temporary file and return it
    _, buffer = cv2.imencode('.jpg', output_image)
    response_image = base64.b64encode(buffer).decode('utf-8')
    return jsonify({"output_image": response_image})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
