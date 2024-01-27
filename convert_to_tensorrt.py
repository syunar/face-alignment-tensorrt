import face_alignment
from PIL import Image
import numpy as np
import torch
import time

def benchmark(model, input_shape=(1, 3, 256, 256), dtype='fp32', nwarmup=50, nruns=10000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype=='fp16':
        input_data = input_data.half()
        
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%10==0:
                print('Iteration %d/%d, ave batch time %.2f ms'%(i, nruns, np.mean(timings)*100))
                print(f'per image: {np.mean(timings)*100 / input_shape[0]:.2f} ms')

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device="cuda")
input = np.array(Image.open('test/assets/aflw-test.jpg'))
input_t = torch.from_numpy(input)
batch_size = 1
input_t = input_t.unsqueeze(0).repeat(batch_size, 1, 1, 1).permute(0, 3, 1, 2)
print(input_t.shape)
preds = fa.get_landmarks_from_batch(input_t)
print("original fa.face_detector.face_detector")
benchmark(fa.face_detector.face_detector, input_shape=(1, 3, 256 ,256), nruns=100)

example = torch.rand([1, 3, 256, 256]).to("cuda")
traced_script_module = torch.jit.trace(fa.face_detector.face_detector.eval(), example, strict=False)

import torch
import torch_tensorrt
trt_ts_module = torch_tensorrt.compile(
        traced_script_module,
        inputs = [torch_tensorrt.Input(example.shape, dtype=torch.float)],
        enabled_precisions = {torch.float},
        truncate_long_and_double = True,
        # require_full_compilation=True,
)

print("tensorrt fa.face_detector.face_detector")
benchmark(trt_ts_module, input_shape=(1, 3, 256 ,256), nruns=100)
torch.jit.save(trt_ts_module, "face_detector_fp16.ts")


# import torch
# import torch_tensorrt
# trt_ts_module = torch_tensorrt.compile(
#         traced_script_module,
#         inputs = [torch_tensorrt.Input(min_shape=[1, 3, 256, 256],
#                                        opt_shape=[16, 3, 256, 256],
#                                        max_shape=[32, 3, 256, 256], dtype=torch.float)],
#         enabled_precisions = {torch.half},
#         truncate_long_and_double = True,
#         # require_full_compilation=True,
# )

# benchmark(trt_ts_module, input_shape=(32, 3, 256 ,256), nruns=100)


# torch.jit.save(trt_ts_module, "face_detector_fp16_dynamicshape.ts")




# ##

# fa.face_alignment_net


example = torch.rand([1, 3, 256, 256]).to("cuda")

print("original fa.face_alignment_net")
benchmark(fa.face_alignment_net, input_shape=(1, 3, 256 ,256), nruns=100)

traced_script_module = torch.jit.trace(fa.face_alignment_net.eval(), example, strict=False)

import torch
import torch_tensorrt
trt_ts_module = torch_tensorrt.compile(
        traced_script_module,
        inputs = [torch_tensorrt.Input([1, 3, 256, 256], dtype=torch.float)],
        enabled_precisions = {torch.half},
        truncate_long_and_double = True,
        # require_full_compilation=True,
)
print("tensorrt fa.face_alignment_net")
benchmark(trt_ts_module, input_shape=(1, 3, 256 ,256), nruns=100)
torch.jit.save(trt_ts_module, "face_alignment_fp16.ts")



