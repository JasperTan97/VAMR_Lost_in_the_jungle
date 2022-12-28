# VAMR_Lost_in_the_jungle
As part of the mini-project for the [Vision Algorithms for Mobile Robotics](https://rpg.ifi.uzh.ch/teaching.html) (VAMR) course at UZH taught by Prof Davide Scaramuzza, we (Abhiram, Adarsh, Jasper and [TianYi](https://github.com/tianyilim)) implement a monocular Visual Odometry (VO) pipeline on the KITTI, Malaga and Parking datasets.

## Setup
1. Clone this repository.
2. Navigate to the root folder of this repo.
3. Run `setup.sh`. It should:
   1. Setup a `conda` environment, `vamr_proj`, with `requirements.txt` installed.
   2. Download the datasets from the VAMR course websites.
4. Try out our examples (**TODO**).

## Repository Structure
- `data`: Our example datasets. These comprise a list of images along with the corresponding `K` camera intrinsic matrics.
  - `Parking` dataset
  - `KITTI` dataset
  - `Malaga` dataset
- `code`: Our functions for VO
- `_examples`: Testing/demo code. **SHOULD BE REMOVED BEFORE MAKING REPO PUBLIC**
- `main.py`: Our main function. Run this to see the output.

## Task allocation
### TY
- [ ] Investigate OpenCV functions
  - [ ] PNP RANSAC
  - [x] Find Fundemental Matrix (5/8 point RANSAC)
    - [x] Extract R,T
- [x] Open and process dataset
  - [x] Get `K` from it (so far only working on Parking dataset)
- [X] Triangulation 

### `J`a`s`p`e`r
- [X] Implementation of `VO_State`
- [X] Gonna do KLT LOL
 - I will presume it will work
- [X] Feature detection

### Abhiram
- Bootstrapping
    - [ ] Feature matching
    - [ ] 5/8 point + RANSAC

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
