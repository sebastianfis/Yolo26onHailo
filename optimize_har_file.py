from hailo_sdk_client import ClientRunner
import numpy as np

'''
NOTE: This script is to be run on Linux with hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64 installed

End nodes mapped from original model: 
 '/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv', 
 '/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv', 
 '/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv', 
 '/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv', 
 '/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv', 
 '/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv'.
'''
model_name = 'yolo26m_hailo_model'

runner = ClientRunner(hw_arch='hailo8l', har= model_name + '.har')

model_script_commands = [
    "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n",
    "resize_input1 = resize(resize_shapes=[640,640])\n",

    "change_output_activation(conv74, sigmoid)\n",
    "change_output_activation(conv90, sigmoid)\n",
    "change_output_activation(conv104, sigmoid)\n",
    "quantization_param([conv74, conv90, conv104], force_range_out=[0.0, 1.0])\n",

    # Highest safe level for 16 GB
    "model_optimization_flavor(optimization_level=2, compression_level=2)\n",
    "performance_param(compiler_optimization_level=max)\n",
    #"adaround(policy=enabled)\n",   # let SDK auto-pick small dataset
    "post_quantization_optimization(finetune, policy=enabled, epochs=6)\n",
]


calib_dataset = np.load("calib_set.npy")

runner.load_model_script("".join(model_script_commands))
runner.optimize(calib_dataset)

# Let's save the runner's state to a Quantized HAR
quantized_model_har_path = f"{model_name}_quantized_model.har"
runner.save_har(quantized_model_har_path)


