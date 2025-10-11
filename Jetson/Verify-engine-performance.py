import tensorrt as trt


with open("yolov11_ghost.engine", "rb") as f:
    engine_data = f.read()
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_data)


context = engine.create_execution_context()


inputs, outputs, bindings = [], [], []
for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
    dtype = trt.nptype(engine.get_binding_dtype(binding))

    device_mem = cuda.mem_alloc(size * dtype.itemsize)
    bindings.append(int(device_mem))
    if engine.binding_is_input(binding):
        inputs.append({"device": device_mem, "dtype": dtype, "shape": engine.get_binding_shape(binding)})
    else:
        outputs.append({"device": device_mem, "dtype": dtype, "shape": engine.get_binding_shape(binding)})


stream = cuda.Stream()
cuda.memcpy_htod_async(inputs[0]["device"], input_data, stream)
context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
cuda.memcpy_dtoh_async(output_data, outputs[0]["device"], stream)
stream.synchronize()