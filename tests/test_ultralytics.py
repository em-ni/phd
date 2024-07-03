"""
From gpt:
YOLO Model	        GPU Memory Required for Training	GPU Memory Required for Inference
YOLOv1	            ~4 GB	                            ~1 GB
YOLOv2	            ~4 GB	                            ~1 GB
YOLOv3	            6-8 GB	                            1-2 GB
YOLOv4	            8-12 GB	                            2-4 GB
YOLOv4-tiny	        4-6 GB	                            ~1 GB
YOLOv5 Small	    6-8 GB	                            ~2 GB
YOLOv5 Medium	    8-12 GB	                            2-4 GB
YOLOv5 Large	    12-16 GB	                        4-6 GB
YOLOv5 X-Large	    16+ GB	                            6-8 GB
YOLOv6	            12-24 GB (depending on version)	    4-8 GB (depending on version)
YOLOv7	            12-24 GB (depending on version) 	4-8 GB (depending on version)
YOLOv8	            12-24 GB (depending on version)	    4-8 GB (depending on version)
YOLOv9	            16-32 GB (depending on version)	    6-10 GB (depending on version)
YOLOv10	            16-32 GB (depending on version)	    6-10 GB (depending on version)
"""

"""
From ultralytics website:
Model	        Params (M)	FLOPs (G)	APval (%)	Latency (ms)	Latency (Forward) (ms)
YOLOv6-3.0-N	4.7     	11.4	    37.0	    2.69	        1.76
Gold-YOLO-N	    5.6	        12.1	    39.6	    2.92	        1.82
YOLOv8-N	    3.2	        8.7	        37.3	    6.16	        1.77
YOLOv10-N	    2.3	        6.7	        39.5	    1.84	        1.79
YOLOv6-3.0-S	18.5	    45.3	    44.3	    3.42	        2.35
Gold-YOLO-S	    21.5	    46.0	    45.4	    3.82	        2.73
YOLOv8-S	    11.2	    28.6	    44.9	    7.07	        2.33
YOLOv10-S	    7.2	        21.6	    46.8	    2.49	        2.39
RT-DETR-R18	    20.0	    60.0	    46.5	    4.58        	4.49
YOLOv6-3.0-M	34.9	    85.8	    49.1	    5.63	        4.56
Gold-YOLO-M	    41.3	    87.5	    49.8	    6.38	        5.45
YOLOv8-M	    25.9	    78.9	    50.6	    9.50        	5.09
YOLOv10-M	    15.4	    59.1	    51.3	    4.74        	4.63
YOLOv6-3.0-L	59.6	    150.7   	51.8    	9.02	        7.90
Gold-YOLO-L	    75.1	    151.7   	51.8    	10.65       	9.78
YOLOv8-L	    43.7	    165.2   	52.9    	12.39	        8.06
RT-DETR-R50	    42.0	    136.0   	53.1    	9.20	        9.07
YOLOv10-L	    24.4	    120.3   	53.4    	7.28        	7.21
YOLOv8-X	    68.2	    257.8   	53.9    	16.86	        12.83
RT-DETR-R101	76.0	    259.0   	54.3    	13.71	        13.58
YOLOv10-X	    29.5	    160.4   	54.4    	10.70	        10.60
"""

import cv2
from PIL import Image

from ultralytics import YOLO

# model = YOLO("yolov10n.pt")
# model = YOLO("yolov10s.pt")
# model = YOLO("yolov10m.pt")
# model = YOLO("yolov10b.pt")
# model = YOLO("yolov10l.pt")
model = YOLO("yolov10x.pt")

# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
# results = model.predict(source="0")
# results = model.predict(source="folder", show=True)  # Display preds. Accepts all YOLO predict arguments

# from PIL
# im1 = Image.open("bus.jpg")
# results = model.predict(source=im1, save=True)  # save plotted images

# from ndarray
im2 = cv2.imread("bus.jpg")
results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

# from list of PIL/ndarray
# results = model.predict(source=[im1, im2])