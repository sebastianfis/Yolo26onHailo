from ultralytics import YOLO


def export_onnx(name="yolo26m"):
    model = YOLO(name + ".pt")
    model.export(format='onnx', end2end=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="yolo26m")
    args = parser.parse_args()
    export_onnx(name=args.name)
