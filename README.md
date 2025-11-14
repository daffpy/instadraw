# InstaDraw

## what is InstaDraw?

It is a viral computer vision image tracing project made with OpenCV and PyAutoGUI for Instagram's latest doodle feature update.
It uses a combination contouring and thresholding of a custom image to simulate pixel-to-pixel image drawing on Instagram app

### using OpenCV contouring:
https://github.com/user-attachments/assets/bdc5c008-a7d5-436e-899a-a021be8582d9

_this video reaches 2.1M views on tiktok [(@browhateveridc69)](https://vt.tiktok.com/ZSy3UjAhU/)_

### using OpenCV thresholding:

https://github.com/user-attachments/assets/30aae85d-bf52-4f44-aaa5-39ce874613dc



## what are the requirements?

It requires [`python3.13`](https://www.python.org/downloads/) to be installed on your computer and all of its library or module listed on `requirements.txt`. It also requires a use of emulator for example: BlueStack or built-in Iphone Mirroring feature on Mac.

## Setup

1. Simply prompt on your terminal/command prompt `pip install -r requirements.txt` to install all of the required library
2. Modify all of pyautogui coordinate by assigning your emulator window's coordinate, you could check the coordinate by calling `getCoord()` method, and it will reveal the current position of the cursor
3. Modify the image path if you intend to use custom images (note: every image might need a different level of thresholding and tweaking)
4. use the `scale_image()` method to scale the image into a smaller resolution to fit your emulator window
5. use `drawManual()` method for a contour-like result and `drawManualThresh()` for a full but black-and-white result only


