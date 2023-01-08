# VAMR_Lost_in_the_jungle

As part of the mini-project for the [Vision Algorithms for Mobile Robotics](https://rpg.ifi.uzh.ch/teaching.html) (VAMR) course at UZH taught by Prof Davide Scaramuzza, we (Abhiram, Adarsh, Jasper and [TianYi](https://github.com/tianyilim)) implement a monocular Visual Odometry (VO) pipeline on the KITTI, Malaga and Parking datasets.

Our report is [here](Visual_Odometry_Pipeline_Report.pdf)

## Setup
1. Clone this repository.
2. Navigate to the root folder of this repo.
3. Run `setup.sh`. It should:
   1. Setup a `conda` environment, `vamr_proj`, with `requirements.txt` installed.
   2. Download the datasets from the VAMR course websites.
4. Try out our examples.

## Repository Structure
- `data`: Our example datasets. These comprise a list of images along with the corresponding `K` camera intrinsic matrics.
  - `Parking` dataset
  - `KITTI` dataset
  - `Malaga` dataset
- `code`: Our functions for VO
- `Notes`: Some notes we made along the way.
- `main.py`: Our main function. Run this to see the output.

## Running our code
To run our code, just run `python3 main.py` in the root folder of this repository.

![Viz](Notes/Code%20Illus.png)

This will display three plots.
1. The top plot shows the image, along with an overlay of the keypoints being tracked as red crosses.
2. The bottom left plot shows a history of the poses computed through VO. The red dot is the most recent pose.
3. The bottom right plot shows the number of keypoints that have been tracked in the past 20 frames.

### Dataset
To change between datasets, set the `DATASET` variable in `main.py`. It is a string accepting `parking`, `kitti`, or `malaga`.

### Plotting pointcloud
Our implementation uses 2D-2D correspondences for VO, and therefore does not provide a pointcloud by default.

To visualise a pointcloud, set the `PLOT_POINTCLOUD` variable in `main.py` to `True`. It will also visualise a point-cloud triangulated from the last few poses.

![Pcl Viz](Notes/Pointcloud%20Illus.png)
This will display four plots.
1. The plot on the top left shows a history of the poses computed through VO. The red dot is the most recent pose.
2. The plot on the top right shows the image, along with an overlay of the keypoints being tracked as red crosses.
3. The plot on the bottom left shows the number of keypoints that have been tracked in the past 20 frames.
4. The plot on the bottom right shows: 
   1. the past poses as blue arrows,
   2. the current pose as a yellow arrow,
   3. the base pose for triangulation as a red arrow,
   4. the pose calculated from triangulating poses from the current pose and base pose as a large purple arrow,
   5. and the pointcloud as green crosses.

## Video recordings
Go to the Polybox link [here](https://polybox.ethz.ch/index.php/s/089LXzUeORKMswT) to view our recorded videos.
