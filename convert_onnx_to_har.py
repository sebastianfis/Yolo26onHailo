from hailo_sdk_client import ClientRunner

'''
NOTE: This script is to be run on Linux with hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64 installed!

End nodes mapped from original model (Checked for s and m model. Might have to be changed for different model sizes!): 
        "/model.23/cv2.0/cv2.0.2/Conv",
        "/model.23/cv3.0/cv3.0.2/Conv",
        "/model.23/cv2.1/cv2.1.2/Conv",
        "/model.23/cv3.1/cv3.1.2/Conv",
        "/model.23/cv2.2/cv2.2.2/Conv",
        "/model.23/cv3.2/cv3.2.2/Conv"
'''


def convert_onnx_to_har(name="yolo26m", hw_arch="hailo8l"):
    # Initialize the ClientRunner
    onnx_path = name + ".onnx"
    runner = ClientRunner(hw_arch=hw_arch)

    # Use the recommended end node names for translation
    end_node_names = [
        "/model.23/cv2.0/cv2.0.2/Conv",
        "/model.23/cv3.0/cv3.0.2/Conv",
        "/model.23/cv2.1/cv2.1.2/Conv",
        "/model.23/cv3.1/cv3.1.2/Conv",
        "/model.23/cv2.2/cv2.2.2/Conv",
        "/model.23/cv3.2/cv3.2.2/Conv",
    ]

    try:
        # Translate the ONNX model to Hailo's format
        hn, npz = runner.translate_onnx_model(
            onnx_path,
            name,
            end_node_names=end_node_names,
            net_input_shapes={"images": [1, 3, 640, 640]},  # Adjust input shapes if needed
        )
        print("Model translation successful.")
    except Exception as e:
        print(f"Error during model translation: {e}")
        raise

    # Save the Hailo model HAR file
    hailo_model_har_name = f"{name}_hailo_model.har"
    try:
        runner.save_har(hailo_model_har_name)
        print(f"HAR file saved as: {hailo_model_har_name}")
    except Exception as e:
        print(f"Error saving HAR file: {e}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="yolo26m")
    parser.add_argument('--hwarch', type=str, default="hailo8l")
    args = parser.parse_args()
    convert_onnx_to_har(name=args.name, hw_arch=args.hwarch)
