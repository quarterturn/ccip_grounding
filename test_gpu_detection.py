import onnxruntime as ort
try:
    providers = ort.get_available_providers()
    print(f"Available Providers: {providers}")
    if "CUDAExecutionProvider" in providers:
        print("SUCCESS: CUDAExecutionProvider is available!")
        # Try to actually initialize a session to trigger the cublasLt load
        sess = ort.InferenceSession(None, providers=["CUDAExecutionProvider"])
    else:
        print("FAILURE: CUDAExecutionProvider not found in providers list.")
except Exception as e:
    print(f"CRASHED: {e}")
