import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import random
import numpy as np
import cv2
import io
from pytorch_lightning import seed_everything
import einops
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import config

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)

# Load Model and other dependencies
apply_canny = CannyDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/saved_model_canny.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    results = []
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results


def process_image(image):
    resized_img = cv2.resize(image, (256, 256))
    if resized_img.dtype != np.uint8:
        resized_img = (resized_img * 255).astype(np.uint8) if resized_img.max() <= 1 else resized_img.astype(np.uint8)
    return resized_img


@app.route('/process', methods=['POST'])
def process_request():
    # Parse input data from the frontend
    data = request.json
    image_data = data['image']  # Base64 string
    prompt = data['prompt']
    a_prompt = data.get('a_prompt', '')
    n_prompt = data.get('n_prompt', '')
    num_samples = data.get('num_samples', 1)
    image_resolution = data.get('image_resolution', 256)
    ddim_steps = data.get('ddim_steps', 50)
    guess_mode = data.get('guess_mode', False)
    strength = data.get('strength', 1.0)
    scale = data.get('scale', 7.5)
    seed = data.get('seed', -1)
    eta = data.get('eta', 0.0)
    low_threshold = data.get('low_threshold', 50)
    high_threshold = data.get('high_threshold', 200)

    # Convert base64 image to OpenCV format
    image_data = np.frombuffer(io.BytesIO(base64.b64decode(image_data)).read(), np.uint8)
    input_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    # Process image with the model
    output_image = process(
        input_image=input_image,
        prompt=prompt,
        a_prompt=a_prompt,
        n_prompt=n_prompt,
        num_samples=num_samples,
        image_resolution=image_resolution,
        ddim_steps=ddim_steps,
        guess_mode=guess_mode,
        strength=strength,
        scale=scale,
        seed=seed,
        eta=eta,
        low_threshold=low_threshold,
        high_threshold=high_threshold
    )[0]  # Get the first result

    # Save the output image as a temporary file and return it
    _, buffer = cv2.imencode('.png', output_image)
    response_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({"output_image": response_image})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
