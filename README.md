# COMP0090CW2
pip libraries that need to be installed:
  - h5py
  - IPython
  - cv2
  - torchviz

Code should be run in order of:

  - For original MTL Model:
    - networkaux.py
    - trainaux.py
    - networktarg.py
    - traintarg.py
      - This will result in model_mtl.pth
    
    
  - For Baseline Model:
    - networkbase.py
    - trainbase.py
      - This will result in model_sgl.pth
    
  - For Ablated MTL Model:
    - networkab.py
    - trainab.py
      - This will result in model_mtl2.pth
