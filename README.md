<h2 align="center">
  YOLO26 on HAILO
</h2>


---
 This repository presents a workflow to compile the YOLO26 
 models for object detection trained on the COCO dataset to 
 .hef files to be used with the HAILO AI accelerators. It 
 uses the hailo dataflow compiler V3.33
  
## ⚡ Disclaimer
**This Repo is currently still work in progress. 
I have been able to use this workflow to successully generate 
a .hef file from the YOLO26m model, but did not get around to 
test its performance, or benchmark it.** 

**Also, I don't have a 
compatible GPU right now, so I am limited to rather low 
optimization settings and could not test possible issues/tweaks 
with higher opt. settings.**

**I did my best to optimize the settings for 
a solid perfomrance/FPS compromise. Feel free to play aroud with 
the settings, I am sure, there is still a lot of optimization 
possible, especially if you have a CUDA compatible GPU. 
But this is a hobby project for me, so I will not be able 
to provide support**

# 🛠️ Workflow

## 1 Data preperation
Download COCO2017 from [COCO](https://cocodataset.org/#download). 
Then run 

```shell
python3 generate_calib_dataset.py 
--images_path <PATH TO COCO TRAINING IMAGES FOLDER>
--annotations_path <PATH TO COCO TRAIN ANNOTATIONS FILE>
```
This will generate a .npy file (default `calib_set.npy`), that contains 1024 
images (customizable with `--target_size <IMAGE NUMBER>`) to be used for 
calibration and PQT. The image choice will be class balanced to improve 
detection quality across all classes.

## 2 Export YOLO26 to ONNX
Run 

```shell
python3 create_yolo_onnx 
--name yolo26<MODELSIZE, default=m>
```
This will use the ultralytics package to download the respective YOLO26 model and 
export it to .onnx file format. 

Run
```shell
python3 generate_nms_configuration.py
```
This will create a json fle with the configuration for the NMS, that will be added 
to the model later on

<details>
<summary> If the next step fails:</summary>

Check the .onnx file with [Netron](https://netron.app/). 
Look for the final convolution nodes in each line. These are the nodes, 
that later represent the NMS input. 

**If the node names are not:**
```
/model.23/cv2.0/cv2.0.2/Conv
/model.23/cv3.0/cv3.0.2/Conv
/model.23/cv2.1/cv2.1.2/Conv
/model.23/cv3.1/cv3.1.2/Conv
/model.23/cv2.2/cv2.2.2/Conv
/model.23/cv3.2/cv3.2.2/Conv
```
Adapt `end_node_names` in `convert_onnx_to_har.py` 
with the correct node names. 
I checked the names with m and s model and at least for those 
two models the node names were equal.

</details>

## 3 Convert ONNX to HAR
Run
```shell
python3 convert_onnx_to_har.py 
--name <YOLO ONNX FILE NAME, default="yolo26m">
--hwarch <TARGET ARCHITECTURE, default="hailo8l">
```
This will  convert the .onnx to the HAILO archive file format using the 
Hailo DFC (V3.33 is required).

## 4 Model quantization
Run
```shell
python3 optimize_har_file.py 
--name <HAR FILE NAME, default="yolo26m_hailo_model">
--hwarch <TARGET ARCHITECTURE, default="hailo8l">
--opt_level <OPTIMIZATION LEVEL, default="2">
--comp_level <COMPRESSION LEVEL, default="3">
--pqt_epochs <EPOCHS TO RUN DURING PQT, default="8">
--pqt_lr <PQT LEARNING RATE>, default="1e-5"
```
This will start the compression and optimization on the .har file, followed by
PQT finetuning.

Note, that I have not been able to test optimization level > 2, as this requires 
a compatible GPU, to finish in a decent timeframe. 

<details>
<summary> If the next step fails:</summary>

Check the .onnx file with [Netron](https://netron.app/). 
Look for the final convolution nodes in each line. These are the nodes, 
that later represent the NMS input. 

**For m and s models at least, the node names are:**
```
/model.23/cv2.0/cv2.0.2/Conv
/model.23/cv3.0/cv3.0.2/Conv
/model.23/cv2.1/cv2.1.2/Conv
/model.23/cv3.1/cv3.1.2/Conv
/model.23/cv2.2/cv2.2.2/Conv
/model.23/cv3.2/cv3.2.2/Conv
```

Run 
```shell
python3 generate_yolo_layer_report.py 
--name <HAR FILE NAME, default="yolo26m_hailo_model">
```
Check the output nodes by comparing the "original node names" to 
those of the .onnx file. Adapt `"reg_layer"`/`cls_layer` pairs 
in `nms_layer_config` of `generate_nms_configuration.py` with 
the correct node names. I checked the names with m and s model 
and at least for those two models the node names were equal.
</details>


## 5 Export .hef file
Run
```shell
python3 export_hef_file.py 
--name < QUANTIZED MODEL HAR FILE, default="yolo26m_hailo_model_quantized_model.har">
--hwarch <TARGET ARCHITECTURE, default="hailo8l">
```
This will add nms to the model output and export it to a hef file, 
that can be run on the HAILO AI accelerators.