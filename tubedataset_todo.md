
## Base Function

+ TubeDataset building
  + [ ] Support UCF101-24
  + [ ] Support JHMDB
  + [ ] Support loading preload_ann_file
+ TubeDataset Pipeline(Due to version changes, some code style and design need to change according to current other augmentation.)
  + [ ] TubeSampleFrames
  + [ ] TubeExtract
  + [ ] TubeFlip
  + [ ] TubePad
  + [ ] TubeResize
  + [ ] TubeDecode
  + [ ] CuboidCrop
  + [ ] FormatTubeShape
+ Tube-wise Spatio-temporal detection metrics
  + source code: ``
  + [x] frame_mean_ap
  + [x] video_mean_ap
  + [x] related-iou calculation
  + [x] related-nms calculation
  + [x] frame_mean_ap_error

## Refactor & Improvement

+ [ ] Reduce duplicate codes for ap/map/iou related codes
+ [ ] Whether to use BboxOverlaps2D to replace iou2d/iou3d
+ [ ] Whether to refactor the input of `frame_mean_ap`
+ [ ] Whether to load pickle with fixed encoding.
+ [ ] TubeDataset evaluation.
+ [ ] Whether to remove `tube_length` in TubeDataset.
