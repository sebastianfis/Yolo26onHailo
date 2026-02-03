import json

'''
End nodes mapped from original model: 
 '/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv', 
 '/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv', 
 '/model.23/one2one_cv2.1/one2one_cv2.1.2/Conv', 
 '/model.23/one2one_cv3.1/one2one_cv3.1.2/Conv', 
 '/model.23/one2one_cv2.2/one2one_cv2.2.2/Conv', 
 '/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv'.
'''

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

# Path to save the updated JSON configuration
output_path = "nms_layer_config.json"

# Save the updated configuration as a JSON file
with open(output_path, "w") as json_file:
    json.dump(nms_layer_config, json_file, indent=4)

print(f"NMS layer configuration saved to {output_path}")
