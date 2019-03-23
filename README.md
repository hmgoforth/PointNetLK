# PointNetLK: Point Cloud Registration using PointNet

### [Video](https://youtu.be/W4N17CO19cQ)

Source Code Author:
Yasuhiro Aoki

### Requires:
* PyTorch 0.4.0 (perhaps, 0.4.1 (the latest) will be OK.) and torchvision
* NumPy
* SciPy
* MatPlotLib
* ModelNet40

### Main files for experiments:
* train_classifier.py: train PointNet classifier (used for transfer learning)
* train_pointlk.py: train PointNet-LK
* generate_rotation.py: generate 6-dim perturbations (rotation and translation) (for testing)
* test_pointlk.py: test PointNet-LK
* test_icp.py: test ICP
* result_stat.py: compute mean errors of above tests

### Examples (Bash shell scripts):
* ex1_train.sh: train PointNet classifier and transfer to PointNet-LK.
* ex1_genrot.sh: generate perturbations for testing
* ex1_test_pointlk.sh: test PointNet-LK
* ex1_test_icp.sh: test ICP
* ex1_result_stat.sh: compute mean errors of above tests

### Citation

'''
@ARTICLE{2019arXiv190305711A,
       author = {{Aoki}, Yasuhiro and {Goforth}, Hunter and
         {Srivatsan}, Rangaprasad Arun and {Lucey}, Simon},
        title = "{PointNetLK: Robust \&amp; Efficient Point Cloud Registration using PointNet}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition},
         year = "2019",
        month = "Mar",
          eid = {arXiv:1903.05711},
        pages = {arXiv:1903.05711},
archivePrefix = {arXiv},
       eprint = {1903.05711},
 primaryClass = {cs.CV}
}
'''
