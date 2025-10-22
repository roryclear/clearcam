<p align="center">
  <img src="images/logo.png" alt="logo" width="100" />
</p>

# clearcam: Turn your RTSP enabled camera or old iPhone into a state of the art AI Security Camera

<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td align="center">
      <a href="https://apps.apple.com/gb/app/clearcam/id6743237694">
        <img src="https://developer.apple.com/assets/elements/badges/download-on-the-app-store.svg"
             alt="Download on the App Store"
             height="50"/>
      </a>
    </td>
    <td align="center">
      <a href="https://play.google.com/store/apps/details?id=com.rors.clearcam">
        <img src="https://play.google.com/intl/en_us/badges/static/images/badges/en_badge_web_generic.png"
             alt="Get it on Google Play"
             height="50"/>
      </a>
    </td>
  </tr>
</table>


<table align="center" cellspacing="0" cellpadding="0" style="border-collapse: collapse;">
  <tr valign="top">
    <td style="padding-right: 10px;">
      <img src="images/server.PNG" alt="Server" width="400" /><br/>
      <img src="images/kg1.jpg" alt="KG1" width="400" /><br/>
      <img src="images/kg2.jpg" alt="KG2" width="400" />
    </td>
    <td>
      <img src="images/front.PNG" alt="Front" width="400" />
    </td>
    <td>
      <img src="images/ios_live.PNG" alt="iOS Live" width="200" />
      <img src="images/ios_events.PNG" alt="iOS Events" width="200" />
    </td>
  </tr>
</table>

### Don't own an RTSP camera yet?
Try it out with this feed: https://webcam.elcat.kg/Too-Ashu_Tunnel_North/index.m3u8 (https://kg.camera)

## video demo:
https://x.com/RoryClear/status/1959249250811785405

## install and run NVR + inference with homebrew (old release)
1. brew tap roryclear/tap
2. brew install clearcam
3. clearcam
4. (optional) enter your Clearcam premium userID (viewable in iOS app) to receive streams and notifications
5. open localhost:8080 in your browser

## run NVR + inference in python from source (recommended)
1. pip install -r requirements.txt
2. python3 clearcam.py
3. (optional) enter your Clearcam premium userID (viewable in iOS app) to receive streams and notifications
4. open localhost:8080 in your browser
- use BEAM=2 python3 clearcam.py for extra performance (wait time on first run)

## python requirements
- ffmpeg (installed on your computer)
- tinygrad
- numpy
- cv2
- scipy

## install iOS App from source
1. git clone https://github.com/roryclear/clearcam.git
2. open ios/clearcam.xcodeproj

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

# Signing Up for Clearcam Premium

## Features
- View your live camera feeds remotely.
- Receive notifications on events (objects/people detected).
- View event clips remotely.
- End-to-end encryption on all data.

## How to Sign Up

Visit [rors.ai](https://www.rors.ai) to sign up, or upgrade to premium in the iOS app.
