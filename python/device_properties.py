import pycuda.driver as cuda
import pycuda.autoinit


def query_cuda_devices():
    device_count = cuda.Device.count()
    for i in range(device_count):
        device = cuda.Device(i)
        attrs = device.get_attributes()

        print(f"Device {i}: {device.name()}")
        print(f"  Compute capability: {device.compute_capability()}")
        print(f"  Total memory: {device.total_memory() / (1024 * 1024)} MB")
        print(f"  Multiprocessors: {attrs[cuda.device_attribute.MULTIPROCESSOR_COUNT]}")
        print(
            f"  Max threads per block: {attrs[cuda.device_attribute.MAX_THREADS_PER_BLOCK]}"
        )
        print(
            f"  Max block dimensions: {attrs[cuda.device_attribute.MAX_BLOCK_DIM_X]} x {attrs[cuda.device_attribute.MAX_BLOCK_DIM_Y]} x {attrs[cuda.device_attribute.MAX_BLOCK_DIM_Z]}"
        )
        print(
            f"  Max grid dimensions: {attrs[cuda.device_attribute.MAX_GRID_DIM_X]} x {attrs[cuda.device_attribute.MAX_GRID_DIM_Y]} x {attrs[cuda.device_attribute.MAX_GRID_DIM_Z]}"
        )
        print(f"  Clock rate: {attrs[cuda.device_attribute.CLOCK_RATE] / 1000} MHz")
        print()


if __name__ == "__main__":
    query_cuda_devices()
