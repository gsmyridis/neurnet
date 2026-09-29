import unittest
from unittest import mock

import mlx.core as mx

from neurnet.device import Device, DeviceType


class DeviceTests(unittest.TestCase):
    def test_device_type_parses_case_insensitively(self) -> None:
        self.assertEqual(DeviceType.from_string("GPU"), DeviceType.GPU)

    def test_cpu_maps_to_torch_and_mlx_cpu(self) -> None:
        device = Device(DeviceType.CPU)

        self.assertTrue(device.is_cpu())
        self.assertFalse(device.is_gpu())
        self.assertEqual(device.to_torch().type, "cpu")
        self.assertEqual(device.to_mlx(), mx.DeviceType.cpu)

    @mock.patch("neurnet.device.mx.device_count", return_value=1)
    def test_gpu_maps_to_mlx_gpu(self, _device_count: mock.Mock) -> None:
        device = Device(DeviceType.GPU)

        self.assertTrue(device.is_gpu())
        self.assertEqual(device.to_mlx(), mx.DeviceType.gpu)

    @mock.patch("neurnet.device.mx.device_count", return_value=0)
    def test_gpu_requires_an_available_mlx_gpu(self, _device_count: mock.Mock) -> None:
        with self.assertRaisesRegex(ValueError, "MLX GPU is not available"):
            Device(DeviceType.GPU).to_mlx()


if __name__ == "__main__":
    unittest.main()
