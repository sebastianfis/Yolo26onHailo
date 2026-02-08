from hailo_sdk_client import ClientRunner
from pprint import pprint

'''
NOTE: This script is to be run on Linux with hailo_dataflow_compiler-3.33.0-py3-none-linux_x86_64 installed!
'''


def generate_yolo_layer_report(har_path="yolo26m_hailo_model.har"):
    # Load the HAR file
    runner = ClientRunner(har=har_path)

    try:
        # Access the HailoNet as an OrderedDict
        hn_dict = runner.get_hn()
        print("Inspecting layers from HailoNet (OrderedDict):")

        # Pretty-print each layer
        for key, value in hn_dict.items():
            print(f"Key: {key}")
            pprint(value)
            print("\n" + "="*80 + "\n")  # Add a separator between layers for clarity

    except Exception as e:
        print(f"Error while inspecting hn_dict: {e}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="yolo26m_hailo_model.har")
    args = parser.parse_args()
    generate_yolo_layer_report(har_path=args.name)
