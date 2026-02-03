from hailo_sdk_client import ClientRunner

# Define the ONNX model path and configuration
onnx_path = "yolo26m.onnx"
onnx_model_name = "yolo26m"
chosen_hw_arch = "hailo8l"  # Specify the target hardware architecture

# Initialize the ClientRunner
runner = ClientRunner(hw_arch=chosen_hw_arch)

# Use the recommended end node names for translation
end_node_names = [
    "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv",
    "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv",
    "/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv",
    "/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv",
    "/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv",
    "/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv",
]

try:
    # Translate the ONNX model to Hailo's format
    hn, npz = runner.translate_onnx_model(
        onnx_path,
        onnx_model_name,
        end_node_names=end_node_names,
        net_input_shapes={"images": [1, 3, 640, 640]},  # Adjust input shapes if needed
    )
    print("Model translation successful.")
except Exception as e:
    print(f"Error during model translation: {e}")
    raise

# Save the Hailo model HAR file
hailo_model_har_name = f"{onnx_model_name}_hailo_model.har"
try:
    runner.save_har(hailo_model_har_name)
    print(f"HAR file saved as: {hailo_model_har_name}")
except Exception as e:
    print(f"Error saving HAR file: {e}")

