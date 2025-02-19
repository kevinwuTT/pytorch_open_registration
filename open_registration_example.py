import torch
from utils.custom_device_mode import ttnn_module, enable_ttnn_device
import ttnn

import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    handlers=[logging.FileHandler("test_fixture.log"), logging.StreamHandler(sys.stdout)],
)

torch.utils.rename_privateuse1_backend('ttnn')

ttnn_device = ttnn_module.custom_device(0)

logging.info("Creating negative ones on cpu")
x1 = torch.neg(torch.ones(32, 32, dtype = torch.bfloat16, device = "cpu"))
print(x1)

logging.info("Transferring to ttnn")
x1 = x1.to(ttnn_device)

logging.info("get underlying ttnn tensor")
x1 = ttnn_module.get_ttnn_tensor(x1)

logging.info("Running abs on ttnn")
x1 = ttnn.abs(x1)

logging.info("calling to_torch")
x1 = ttnn.to_torch(x1)
print(x1)

# logging.info("Running abs using torch.abs")
# x1 = torch.abs(x1)
# print(x1.device)

# logging.info("Copying back to host")
# x1 = x1.to("cpu")
# print(x1)

logging.info("Closing device")
ttnn_module.close_custom_device(ttnn_device)
