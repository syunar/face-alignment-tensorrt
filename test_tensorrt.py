import face_alignment
from face_alignment.api_trt import FaceAlignment_tft
from PIL import Image
import numpy as np
import torch
import time
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

def plot_lm(image, lm):
    # image = Image.fromarray(input)
    lm_pts = lm[0].astype(int)
    draw = ImageDraw.Draw(image)

    # Plot the landmarks on the image
    for i, point in enumerate(lm_pts):
        x, y = point
        # You can customize the radius and color of the points
        radius = 2
        if i == 33 or i == 27 or i == 62:
            color = (255, 255, 0)
        else:
            color = (255, 0, 0)  # Red color, represented as (R, G, B)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    return image

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device="cuda")

input = np.array(Image.open('test/assets/aflw-test.jpg').resize((256,256)))
input_t = torch.from_numpy(input)
batch_size = 1
input_t = input_t.unsqueeze(0).repeat(batch_size, 1, 1, 1).permute(0, 3, 1, 2).float()
print(input_t.shape)
preds = fa.get_landmarks_from_batch(input_t)
start = time.time()
for i in tqdm(range(4500)):
    preds = fa.get_landmarks_from_batch(input_t)
end = time.time()
print("original inference time on 4500 images: ", end-start)


fa_trt = FaceAlignment_tft(device="cuda")
preds_tft = fa_trt.get_landmarks_from_batch(input_t)
print("tensorrt inference time on 4500 images: ",end-start)

print("original result")
original_res = plot_lm(Image.fromarray(input), preds)
print(original_res)
original_res.save("original_result.png")

print("tensorrt result")
tensorrt_res = plot_lm(Image.fromarray(input), preds_tft)
print(tensorrt_res)
tensorrt_res.save("tensorrt_result.png")


# fa_trt.face_detector.face_detector(input_t.to("cuda"))

# import torch
# import torch_tensorrt
# face_detector = torch.jit.load("face_detector_fp16.ts")

# face_detector(torch.rand([16, 3, 256, 256]).to("cuda"))