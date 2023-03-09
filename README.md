# surgical-copilot

# Create environment
1. Follow the official guide to install [tensorflow](https://www.tensorflow.org/install/pip)
2. To test GPU open Anaconda Prompt as admin and type 
```
conda activate tf
python ./test/tf_GPU.py

```
to test with tensorflow or 
```
conda activate tf
python ./test/pt_GPU.py

```
to test with pytorch

3. Install opencv using [conda](https://anaconda.org/conda-forge/opencv)
4. Test cv2 with your webcam
```
python ./test/cv2_cam.py
``` 

# Online images collection
Instructions to collect images while performing ductoscopy
1. Be sure the virtual environment is active and you are in "data collection" folder
2. Edit get_images.py and set labels and number of images
3. Run get_images.py
4. Collected images will be automatically stored in new created folders corresponding to the labels

# Offline images labelling
1. If first use run setup_labelImg.py to clone the repository. <br/>
2. Run set_labels.py
3. Opend folder of the images you want ot label
4. Press "w" to start labelling and CTRL+S to save the labelled image 
