# Example of generating skull .stl from dicom
Be sure you followed setup.md  
1. cd ${HOME}/Desktop/github/sim/script
2. python dicom2stl/dicom2stl.py -t bone -o test/skull.stl dicom2stl/examples/Data/ct_head.nii.gz

# Run sofa python script
1. cd ${HOME}/Desktop/github/sim/script
2. runSofa -l /home/emanuele/Desktop/github/sim/sofa/build/v22.12/lib/libSofaPython3.so ./main.py 