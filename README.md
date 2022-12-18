# VAMR_Lost_in_the_jungle
As part of the mini-project for the [Vision Algorithms for Mobile Robotics](https://rpg.ifi.uzh.ch/teaching.html) (VAMR) course at UZH taught by Prof Davide Scaramuzza, we (Abhiram, Adarsh, Jasper and [TianYi](https://github.com/tianyilim)) implement a monocular Visual Odometry (VO) pipeline on the KITTI, Malaga and Parking datasets.

## Setup
1. Clone this repository.
2. Navigate to the root folder of this repo.
3. Run `setup.sh`. It should:
   1. Setup a `conda` environment, `vamr_proj`, with `requirements.txt` installed.
   2. Download the datasets from the VAMR course websites.
4. Try out our examples (**TODO**).

## Task allocation
### TY
- [ ] Investigate OpenCV functions
  - [ ] PNP RANSAC
  - [ ] Find Fundemental Matrix (5/8 point RANSAC)
    - [ ] Extract R,T
- [x] Open and process dataset
  - [x] Get `K` from it (so far only working on Parking dataset)
- [ ] Implementation of `VO_State`

### `J`a`s`p`e`r
- [ ] Gonna do KLT
    - I will presume it will work

### Abhiram
- Bootstrapping
    - [ ] Feature detection
    - [ ] Feature matching
    - [ ] 5/8 point + RANSAC
    - [ ] Triangulation

### Adarsh
- [ ] firstObservationTracker
- [ ] cameraPoseMappedTo_c
- [ ] TriangulateProMaxPlusUltra
- [ ] Supervise 👀
- [ ] Step out of his comfort zone

## Long (?) in the Future
- [ ] Tidy up code
- [ ] Report
- [ ] Visualise trajectory
- [ ] Other +++ points for 0.5
  - [ ] Collect own dataset
  - [ ] Loop Closure
  - [ ] Analyze trajectory