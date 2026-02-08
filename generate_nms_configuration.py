import json

'''
End nodes mapped from original model: 
        "/model.23/cv2.0/cv2.0.2/Conv",
        "/model.23/cv3.0/cv3.0.2/Conv",
        "/model.23/cv2.1/cv2.1.2/Conv",
        "/model.23/cv3.1/cv3.1.2/Conv",
        "/model.23/cv2.2/cv2.2.2/Conv",
        "/model.23/cv3.2/cv3.2.2/Conv"
'''


def generate_nms_config(output_path="nms_layer_config.json"):
    # Updated NMS layer configuration dictionary
    nms_layer_config = {
        "nms_scores_th": 0.2,
        "nms_iou_th": 0.7,
        "image_dims": [
            640,
            640
        ],
        "max_proposals_per_class": 100,
        "classes": 80,
        "regression_length": 16,
        "background_removal": False,
        "background_removal_index": 0,
        "bbox_decoders": [
            {
                "name": "bbox_decoder71",
                "stride": 8,
                "reg_layer": "conv71",
                "cls_layer": "conv74"
            },
            {
                "name": "bbox_decoder87",
                "stride": 16,
                "reg_layer": "conv87",
                "cls_layer": "conv90"
            },
            {
                "name": "bbox_decoder101",
                "stride": 32,
                "reg_layer": "conv101",
                "cls_layer": "conv104"
            }
        ]
    }

    # Save the updated configuration as a JSON file
    with open(output_path, "w") as json_file:
        json.dump(nms_layer_config, json_file, indent=4)

    print(f"NMS layer configuration saved to {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default="nms_layer_config.json")
    args = parser.parse_args()
    generate_nms_config(output_path=args.output_path)
