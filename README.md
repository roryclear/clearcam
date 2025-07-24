# clearcam
# Turn your RTSP enabled camera or old iPhone into a state of the art AI Security Camera
## Now on the Apple App Store ##
https://apps.apple.com/app/clearcam/id6743237694

<p align="center">
  <img src="front.png" alt="Front" width="300" />
  &nbsp;&nbsp;&nbsp;
  <img src="server.png" alt="Server" width="300" />
</p>

## run DVR in python
1. pip install -r requirements.txt
2. python3 clearcam.py
3. (optional) enter your Clearam premium userID (viewable in iOS app) to receive streams and notifications
4. add your rtsp cameras
- use BEAM=2 python3 clearcam.py for extra performance (wait time on first run)
- use --yolo_size={s, m, l, or x for larger yolov8 variants}

## install ios from source
1. git clone https://github.com/roryclear/clearcam.git
2. open ios/clearcam.xcodeproj

## python requirements
- ffmpeg
- tinygrad
- numpy
- cv2
- scipy
- lap
- cython_bbox

## iOS requirements
- iOS 15 or newer
- iPhone SE (1st gen) or newer (older iPhones *might* work)
- dependencies: NONE!

</br>
<table>
  <tr>
    <td><img src="images/recording.PNG" alt="Screenshot" width="300"/></td>
    <td><img src="images/browser_events.PNG" alt="Screenshot" width="300"/></td>
  </tr>
</table>
<img src="images/browser_playback.PNG" alt="Screenshot"/>

## experimental features
### own notification server
On an event (change in number of detected objects), clearcam will send the video to an IP address of your choice.
### own inference server
Use an external computer to perform object detection over Wi-Fi.
#### requirements:
- python
- tinygrad
- uvicorn
- Shared Wi-Fi network between your phone and computer
1. run [yolov8.py](https://github.com/roryclear/clearcam/blob/main/yolov8.py) on your computer
2. optional: use "nohup python yolo.py &" to prevent sleeping)
3. optional: add s, m, l, x to command to change yolov8 model size from nano.
4. on your phone, turn on "Use Own Inference Server"
5. enter your computer's IP address + port (:6667) e.g http://192.168.1.23:6667
