# A Robust Vision-based Algorithm for Classifying and Tracking Small Orbital Debris Using On-board Optical Cameras
This study develops a vision-based classification and tracking algorithm to address the challenges of in-situ small orbital debris environment tracking, including debris observability and instrument requirements for small debris observation. The algorithm operates in near real-time and is robust under challenging tasks in moving objects tracking such as multiple moving objects, objects with various movement trajectories and speeds, very small or faint objects, and substantial background motion. The performance of the algorithm is optimized and validated using space image data available through simulated environments generated using NASA Marshall Space Flight Center’s Dynamic Star Field Simulator of on-board optical sensors and cameras.

## Algorithms
> To view the PDF version of each image, please click on it.
### Classification
The classification algorithm identifies the bright spots in captured video sequences and classifies them as `objects` (representing orbital debris objects) or `stars`. In cases the algorithm is not able to classify any of the detected spots into one of the object or star classes, that spot will be classified as `unknowns`. The following block-diagram describes the classification algorithm's outline.
[![Classification](assets/algorithms/classification.png "Classification")](assets/algorithms/classification.pdf)
### Tracking
The tracking algorithm tracks the spots which are classified as objects in the previous step. The following block-diagram describes the tracking algorithm's outline.
[![Tracking](assets/algorithms/tracking.png "Tracking")](assets/algorithms/tracking.pdf)

## Resutls
Ten different datasets were used to evaluate the performance of the algorithms are described in the following table. For all the sample videos *Frame size* is 1080 (pixel) Height &times; 1920 (pixel) Width, *Frame rate* 60 frames per second (fps), *Duration* 60 seconds, *Color format* is Grayscale, and the *Source data type* is Floating-point number between 0 and 1. The title of each sample data (*to see the performance of the classifier for that sample data click on the title*) shows the status of the spacecraft when capturing video frames.

| Sample data            | Spots (mean &plusmn; std) | Objects (mean &plusmn; std) |
|------------------------|--------------------------:|----------------------------:|
| [No Rotation](#no-rotation-sampledatarun2)            |        118.3 &plusmn; 2.1 |            5.3 &plusmn; 2.1 |
| [Single Z Axis Rotation](#single-z-axis-rotation-sampledatarun4) |        114.8 &plusmn; 5.4 |            5.3 &plusmn; 2.0 |
| [Slow Z Axis Rotation](#slow-z-axis-rotation-sampledatarun9)   |        118.3 &plusmn; 2.1 |            5.3 &plusmn; 2.1 |
| [Faster Z Axis Rotation](#faster-z-axis-rotation-sampledatarun5) |        109.0 &plusmn; 5.4 |            4.9 &plusmn; 1.7 |
| [Single X Axis Rotation](#single-x-axis-rotation-sampledatarun3) |        117.7 &plusmn; 3.5 |            5.4 &plusmn; 1.9 |
| [Slow X Axis Rotation](#slow-x-axis-rotation-sampledatarun8)   |        119.6 &plusmn; 1.5 |            5.3 &plusmn; 2.0 |
| [Faster X Axis Rotation](#faster-x-axis-rotation-sampledatarun6) |       148.9 &plusmn; 39.5 |            1.1 &plusmn; 1.0 |
| [Full Rotation](#full-rotation-sampledatarun1)          |        114.0 &plusmn; 4.8 |            5.1 &plusmn; 2.3 |
| [Slow Full Rotation](#slow-full-rotation-sampledatarun7)     |        117.8 &plusmn; 3.3 |            5.2 &plusmn; 2.1 |
| [Faster Full Rotation](#faster-full-rotation-sampledatarun10)   |       133.5 &plusmn; 32.6 |            1.1 &plusmn; 0.6 |

> The following figures show the performance of the classification algorithm for the sample input videos. To view the performance of the tracking algorithm, please click on each figure.

### No Rotation (SampleDataRun2)
[![No Rotation](assets/results/SampleDataRun2-d-1-c-1000-tm-1-ta-1-rc-1-ra-0/classification-confusion-matrix-ul-1.png "No Rotation")](https://youtu.be/Dc5PuvFrSg0)
### Single Z Axis Rotation (SampleDataRun4)
[![Single Z Axis Rotation](assets/results/SampleDataRun4-d-1-c-1000-tm-1-ta-1-rc-1-ra-0/classification-confusion-matrix-ul-1.png "Single Z Axis Rotation")](https://youtu.be/MINUTONbF7E)
### Slow Z Axis Rotation (SampleDataRun9)
[![Slow Z Axis Rotation](assets/results/SampleDataRun9-d-1-c-1000-tm-1-ta-1-rc-1-ra-0/classification-confusion-matrix-ul-1.png "Slow Z Axis Rotation")](https://youtu.be/fxTmYhwb43Y)
### Faster Z Axis Rotation (SampleDataRun5)
[![Faster Z Axis Rotation](assets/results/SampleDataRun5-d-1-c-1000-tm-1-ta-1-rc-1-ra-0/classification-confusion-matrix-ul-1.png "Faster Z Axis Rotation")](https://youtu.be/aKHtzXaMWAY)
### Single X Axis Rotation (SampleDataRun3)
[![Single X Axis Rotation](assets/results/SampleDataRun3-d-1-c-1000-tm-1-ta-1-rc-1-ra-0/classification-confusion-matrix-ul-1.png "Single X Axis Rotation")](https://youtu.be/v9ue5KQiNmA)
### Slow X Axis Rotation (SampleDataRun8)
[![Slow X Axis Rotation](assets/results/SampleDataRun8-d-1-c-1000-tm-1-ta-1-rc-1-ra-0/classification-confusion-matrix-ul-1.png "Slow X Axis Rotation")](https://youtu.be/tgQajvqF-38)
### Faster X Axis Rotation (SampleDataRun6)
[![Faster X Axis Rotation](assets/results/SampleDataRun6-d-1-c-1000-tm-1-ta-1-rc-1-ra-0/classification-confusion-matrix-ul-1.png "Faster X Axis Rotation")](https://youtu.be/THvWsD3J4SI)
### Full Rotation (SampleDataRun1)
[![Full Rotation](assets/results/SampleDataRun1-d-1-c-1000-tm-1-ta-1-rc-1-ra-0/classification-confusion-matrix-ul-1.png "Full Rotation")](https://youtu.be/VFNVB4zX9Qk)
### Slow Full Rotation (SampleDataRun7)
[![Slow Full Rotation](assets/results/SampleDataRun7-d-1-c-1000-tm-1-ta-1-rc-1-ra-0/classification-confusion-matrix-ul-1.png "Slow Full Rotation")](https://youtu.be/Pe9l5cb6sLg)
### Faster Full Rotation (SampleDataRun10)
[![Faster Full Rotation](assets/results/SampleDataRun10-d-1-c-1000-tm-1-ta-1-rc-1-ra-0/classification-confusion-matrix-ul-1.png "Faster Full Rotation")](https://youtu.be/sP8S-DTxj2E)

## References
Zamani, Yasin, et al. "A Robust Vision-based Algorithm for Detecting and Classifying Small Orbital Debris Using On-board Optical Cameras." (2019). [View Article](https://ntrs.nasa.gov/search.jsp?R=20190032383)
