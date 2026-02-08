from hailo_sdk_client import ClientRunner
import numpy as np

'''
NOTE: This script is to be run on Linux with hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64 installed!

End nodes mapped from original model: 
        "/model.23/cv2.0/cv2.0.2/Conv",
        "/model.23/cv3.0/cv3.0.2/Conv",
        "/model.23/cv2.1/cv2.1.2/Conv",
        "/model.23/cv3.1/cv3.1.2/Conv",
        "/model.23/cv2.2/cv2.2.2/Conv",
        "/model.23/cv3.2/cv3.2.2/Conv"
'''


def optimize_har_file(name="yolo26m_hailo_model",
                      hw_arch="hailo8l",
                      opt_level="2",
                      comp_level="3",
                      pqt_epochs="12",
                      pqt_lr="1e-5"):
    model_name = name

    runner = ClientRunner(hw_arch=hw_arch, har=model_name + '.har')

    model_script_commands = [
        "normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n",
        "resize_input1 = resize(resize_shapes=[640,640])\n",
        "model_optimization_config(calibration, calibset_size=1024)\n",
        # "change_output_activation(conv74, sigmoid)\n",
        # "change_output_activation(conv90, sigmoid)\n",
        # "change_output_activation(conv104, sigmoid)\n",
        # "quantization_param([conv74, conv90, conv104], force_range_out=[0.0, 1.0])\n",
        # Highest safe level for 16 GB
        f"model_optimization_flavor(optimization_level={opt_level}, compression_level={comp_level})\n",
        f"post_quantization_optimization(finetune, policy=enabled, epochs={pqt_epochs}, learning_rate={pqt_lr})\n",
    ]

    calib_dataset = np.load("calib_set.npy")

    runner.load_model_script("".join(model_script_commands))
    runner.optimize(calib_dataset)

    # Let's save the runner's state to a Quantized HAR
    quantized_model_har_path = f"{model_name}_quantized_model.har"
    runner.save_har(quantized_model_har_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="yolo26m_hailo_model")
    parser.add_argument('--hwarch', type=str, default="hailo8l")
    parser.add_argument('--opt_level', type=str, default="2")
    parser.add_argument('--comp_level', type=str, default="3")
    parser.add_argument('--pqt_epochs', type=str, default="8")
    parser.add_argument('--pqt_lr', type=str, default="1e-5")
    args = parser.parse_args()
    optimize_har_file(name=args.name,
                      hw_arch=args.hwarch,
                      opt_level=args.opt_level,
                      comp_level=args.comp_level,
                      pqt_epochs=args.pqt_epochs,
                      pqt_lr=args.pqt_lr)

