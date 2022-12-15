# VAMR_Lost_in_the_jungle
Help Dora find her way out using visual odometry

Instructions for where to store data:
Files are in ./data/{parking, kitti, malaga-urban-dataset-extract-07}/...

## Task allocation
### TY
- [ ] Investigate OpenCV functions
  - [ ] PNP RANSAC
  - [ ] Find Fundemental Matrix (5/8 point RANSAC)
    - [ ] Extract R,T
- [ ] Open and process dataset
  - [ ] Get `K` from 
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