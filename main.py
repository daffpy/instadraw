import cv2
import numpy as np
import pyautogui
import time
import os

img = cv2.imread('source17.jpg')

img = cv2.resize(img, None, fx=0.35, fy=0.35)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 1)  

v = np.median(gray)
lower = int(max(0, 0.66 * v))
upper = int(min(255, 1.33 * v))

edges = cv2.Canny(gray, 100,200)

_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def scale_image():
    input_folder = "bad_apple_test"     # folder with original images
    output_folder = "bad_apple_test_scaled"   # folder to save scaled images
    scale = 0.9                       # shrink to 50%

    # create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Skipped unreadable file: {filename}")
                continue

            # resize
            resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # save to output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resized)
            print(f"✅ Scaled and saved: {filename}")

    print("🎯 Done! All images resized to 50%.")


def resample_contour_equal(contour, spacing=4):
    """Resample contour so all points are evenly spaced by `spacing` pixels."""
    contour = contour.reshape(-1, 2)
    # compute distances between consecutive points
    deltas = np.diff(contour, axis=0)
    dists = np.sqrt((deltas ** 2).sum(axis=1))
    cumdist = np.insert(np.cumsum(dists), 0, 0)
    total_length = cumdist[-1]

    # generate uniform distances
    new_dist = np.arange(0, total_length, spacing)
    if len(new_dist) < 2:
        return contour  # too short, skip

    # interpolate x and y along uniform distances
    x_new = np.interp(new_dist, cumdist, contour[:, 0])
    y_new = np.interp(new_dist, cumdist, contour[:, 1])

    return np.stack((x_new, y_new), axis=1).astype(np.int32)

def getContour():
    print(contours)

def simplify_contour(contour, epsilon_ratio=0.001):
    """Reduce the number of points while keeping shape."""
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)

def getSize(contours, episilon_ratio):
    size = 0
    for c in contours:
        contour = simplify_contour(c, epsilon_ratio=episilon_ratio)
        size += len(contour)

    print(f"epsilon ratio: {episilon_ratio}, array size: {size}")

def getCoord():
    while(True):
        print(pyautogui.position())

def drawManual(countours,initial=True):
    scale = 1       
    offset_x = 10  
    offset_y = 220 #pojok kiri
    i = 0
    phase = 0
    time.sleep(3)
    for contour in contours:
        contour = simplify_contour(contour, epsilon_ratio=0.0006)
        contour = contour.reshape(-1,2).astype(int)
        x0, y0 = contour[0].tolist()
        x0 = int(x0 * scale + offset_x)
        y0 = int(y0 * scale + offset_y)
        pyautogui.moveTo(x0, y0)
        pyautogui.mouseDown()
        for x,y in contour.tolist():
            if initial == True:
                if i % 1500 == 0:
                    phase += 1
                    print(f"about to run phase {phase} submit first")
                    if phase > 1:
                        input()
            x = int(x * scale + offset_x)
            y = int(y * scale + offset_y)
            pyautogui.dragTo(x,y,duration=0.001,button='left')
            #print(x,y)
            print(i)
            i += 1
        print('done contour')
        pyautogui.mouseUp()
        time.sleep(0.3)


def drawOpenCV(contours):
    canvas = np.ones_like(img) * 255  # same size, white background
    for contour in contours:
        for i in range(len(contour) - 1):
            pt1 = tuple(contour[i][0])
            pt2 = tuple(contour[i + 1][0])
            cv2.drawContours(canvas, [contour], -1, (0,0,0), thickness=-1)
            cv2.imshow("Tracing", canvas)
        cv2.waitKey(10)

    cv2.imshow("Original", img)
    cv2.imshow("Traced Drawing", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def fill_with_pyautogui(contour, ignore_mask=None, scale=1.0, offset=(0, 0)):
    scale = 1

    # iphone mirroring
    # offset_x, offset_y = 40, 260

    offset_x, offset_y = offset
    print(offset_x,offset_y)

    # bluestack
    # offset_x, offset_y = 60, 254

    # Get dynamic mask size
    h, w = ignore_mask.shape if ignore_mask is not None else (126, 168)
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw this contour as white on mask
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

    for y in range(0, h, 4):
        xs = np.where(mask[y] == 255)[0]
        if len(xs) < 2:
            continue

        # If we have an ignore mask (apple area), remove any xs that overlap
        if ignore_mask is not None:
            ignore_xs = np.where(ignore_mask[y] == 255)[0]
            xs = np.setdiff1d(xs, ignore_xs)  # skip apple pixels

        # Find gaps in remaining white
        if len(xs) < 2:
            continue

        gaps = np.where(np.diff(xs) > 1)[0]
        start_idx = 0
        for gap in np.append(gaps, len(xs) - 1):
            x_start = xs[start_idx]
            x_end = xs[gap]
            pyautogui.moveTo(x_start * scale + offset_x, y * scale + offset_y)
            pyautogui.mouseDown(button='left')
            pyautogui.moveTo(x_end * scale + offset_x, y * scale + offset_y)
            pyautogui.mouseUp(button='left')
            start_idx = gap + 1


def image_extraction(filename):
    img = cv2.imread(f"bad_apple_test_scaled/{filename}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh_inv = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours_black, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    _, white_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    _, ignore_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    mask = np.ones_like(gray, dtype=np.uint8) * 255

    cv2.drawContours(mask, contours_black, -1, 0, thickness=-1)

    contours, hierarchy = cv2.findContours(white_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("Gray",gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return contours, hierarchy, ignore_mask


def drawManualThresh():
    spawnpoint = False
    pyautogui.click(x=275, y=675)

    # pyautogui.click(x=345, y=702) 
    contours=[]
    hierarchy = []

    folder_path = 'bad_apple_scaled'
    time.sleep(2)

#output_0400.jpg

    for frame_idx, entry_name in enumerate(sorted(os.listdir(folder_path))):
        if entry_name != "output_1885.jpg" and spawnpoint is False:
            continue
        else:
            spawnpoint = True
        if frame_idx % 3 != 0:
            continue
        contours, hierarchy, ignore_mask = image_extraction(entry_name)
        if hierarchy is None:
            continue
        start = time.time()

        # (+)
        pyautogui.click(x=275, y=675)

        # pyautogui.click(x=345, y=702) 


        time.sleep(0.2)

        # draw
        pyautogui.click(x=238, y=540)

        # pyautogui.click(x=284, y=634)


        time.sleep(0.2)

        #move to resize
        # pyautogui.moveTo(18,450)
        # pyautogui.mouseDown(button='left')
        # pyautogui.moveTo(18, 505)
        # pyautogui.mouseUp(button='left')


        pyautogui.moveTo(10,450)
        pyautogui.dragTo(10,518, duration=0.05, button='left')

        # pyautogui.moveTo(10,447)
        # pyautogui.dragTo(10,477, duration=0.01, button='left')



        for i, contour in enumerate(contours):
            if hierarchy is None:
                continue
            parent = hierarchy[0][i][3]
            if parent == -1:
                fill_with_pyautogui(contour, ignore_mask)
        time.sleep(0.3)
        #send
        pyautogui.click(x=284, y=678)

        # pyautogui.click(x=354, y=700)



        time.sleep(0.5)
        end = time.time()
        print(f"{entry_name} took {end - start:.4f} secs")


def fill_with_pyautogui_basic(contour, ignore_mask=None, scale=1.0, offset=(0, 0)):
    scale = 1

    # iphone mirroring
    # offset_x, offset_y = 40, 260

    offset_x, offset_y = offset
    print(offset_x,offset_y)

    # bluestack
    # offset_x, offset_y = 60, 254

    # Get dynamic mask size
    h, w = ignore_mask.shape if ignore_mask is not None else (126, 168)
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw this contour as white on mask
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

    for y in range(0, h, 4):
        xs = np.where(mask[y] == 255)[0]
        if len(xs) < 2:
            continue

        # Find gaps in remaining white
        if len(xs) < 2:
            continue

        gaps = np.where(np.diff(xs) > 1)[0]
        start_idx = 0
        for gap in np.append(gaps, len(xs) - 1):
            x_start = xs[start_idx]
            x_end = xs[gap]
            pyautogui.moveTo(x_start * scale + offset_x, y * scale + offset_y)
            pyautogui.mouseDown(button='left')
            pyautogui.moveTo(x_end * scale + offset_x, y * scale + offset_y)
            pyautogui.mouseUp(button='left')
            start_idx = gap + 1


def drawBasic():
    contours, hierarchy, ignore_mask = image_extraction("tes.jpg")
    for i, contour in enumerate(contours):
        if hierarchy is None:
            continue
        parent = hierarchy[0][i][3]
        if parent == -1:
            fill_with_pyautogui(contour, ignore_mask, offset=(280,280))

# drawManual(contours)

# getContour()
# getSize(contours, 0.001)
# getCoord()
# drawOpenCV(contours)

drawManualThresh()
# drawBasic()

# scale_image()

# print(cv2.imread('bad_apple_scaled/output_0001.jpg').shape)


# (+)
# 275 675
# 345 702

# draw
# 238 540
# 284 634

# send
# 284 678
# 354 700

# slider
# 10 450
# 10 505

# 10 447
# 10 477