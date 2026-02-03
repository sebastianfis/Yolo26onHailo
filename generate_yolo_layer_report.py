from hailo_sdk_client import ClientRunner
from pprint import pprint

# Load the HAR file
har_path = "yolo26m_hailo_model.har"

runner = ClientRunner(har=har_path)

try:
    # Access the HailoNet as an OrderedDict
    hn_dict = runner.get_hn()  # Or use runner._hn if get_hn() is unavailable
    print("Inspecting layers from HailoNet (OrderedDict):")

    # Pretty-print each layer
    for key, value in hn_dict.items():
        print(f"Key: {key}")
        pprint(value)
        print("\n" + "="*80 + "\n")  # Add a separator between layers for clarity

except Exception as e:
    print(f"Error while inspecting hn_dict: {e}")