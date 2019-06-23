# PointNetLK: Point Cloud Registration using PointNet

### [Video](https://youtu.be/J2ClR5OZuLc)

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

```
@InProceedings{yaoki2019pointnetlk,
       author = {Aoki, Yasuhiro and Goforth, Hunter and Arun Srivatsan, Rangaprasad and Lucey, Simon},
       title = {PointNetLK: Robust & Efficient Point Cloud Registration Using PointNet},
       booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
       month = {June},
       year = {2019}
}
```
