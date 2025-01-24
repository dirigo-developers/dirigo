import numpy as np
import json
import base64
import time


# Test data: 10,000 float64 values
data = np.random.rand(10_000)

# JSON Default Serialization
start = time.perf_counter()
json_default = json.dumps(data.tolist())
end = time.perf_counter()
print("JSON Default Serialization Time:", end - start)

# JSON Default Deserialization
start = time.perf_counter()
parsed_default = json.loads(json_default)
end = time.perf_counter()
print("JSON Default Deserialization Time:", end - start)

# Base64 Serialization
start = time.perf_counter()
serialized_base64 = base64.b64encode(data.tobytes()).decode('ascii')
json_base64 = json.dumps({'array': serialized_base64, 'dtype': str(data.dtype), 'shape': data.shape})
end = time.perf_counter()
print("Base64 Serialization Time:", end - start)

# Base64 Deserialization
start = time.perf_counter()
loaded = json.loads(json_base64)
decoded = np.frombuffer(base64.b64decode(loaded['array']), dtype=np.float64).reshape(loaded['shape'])
end = time.perf_counter()
print("Base64 Deserialization Time:", end - start)

