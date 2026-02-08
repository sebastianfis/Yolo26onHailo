from hailo_sdk_client import ClientRunner

'''
NOTE: This script is to be run on Linux with hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64 installed!
'''


def export_hef_file(name="yolo26m", hw_arch="hailo8l"):
    runner = ClientRunner(
        hw_arch="hailo8l",
        har="yolo26m_hailo_model_quantized_model.har"
    )

    model_script_commands = [
        "nms_postprocess('nms_layer_config.json', meta_arch=yolov8, engine=cpu)\n",
        "performance_param(compiler_optimization_level=max)\n",
    ]

    runner.load_model_script("".join(model_script_commands))

    hef = runner.compile()

    file_name = "yolo26m.hef"
    with open(file_name, "wb") as f:
        f.write(hef)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="yolo26m")
    parser.add_argument('--hwarch', type=str, default="hailo8l")
    args = parser.parse_args()
    export_hef_file(name=args.name, hw_arch=args.hwarch)
