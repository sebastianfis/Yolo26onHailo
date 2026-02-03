from hailo_sdk_client import ClientRunner
runner = ClientRunner(
    hw_arch="hailo8l",
    har="yolo26m_quantized.har"
)

runner.load_model_script("""
nms_postprocess(
    'nms_layer_config.json',
    meta_arch=yolov8,
    engine=cpu
)
""")

runner.compile()
runner.save_hef("yolo26m.hef")
