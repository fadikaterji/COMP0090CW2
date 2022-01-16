# COMP0090CW2
pip libraries that need to be installed:
  - h5py
  - IPython
  - cv2
  - torchviz

Code should be run in order of:

  - For original MTL Model:
    - trainaux.py
    - traintarg.py
      - This will result in model_mtl.pth
    
    
  - For Baseline Model:
    - trainbase.py
      - This will result in model_sgl.pth
    
  - For Ablated MTL Model:
    - trainab.py
      - This will result in model_mtl2.pth
