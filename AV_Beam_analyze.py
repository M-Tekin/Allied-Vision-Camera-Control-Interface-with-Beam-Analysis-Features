import sys
from typing import Optional
from queue import Queue
import cv2
import matplotlib.pyplot as plt
from vmbpy import *
from skimage import data
from skimage import io
from tkinter import *
from scipy.optimize import curve_fit
import numpy as np


# All frames will either be recorded in this format, or transformed to it before being displayed
opencv_display_format = PixelFormat.Mono8


def print_preamble():
    y = 1
    #print('///////////////////////////////////////////////////')
    #print('/// VmbPy Asynchronous Grab with OpenCV Example ///')
    #print('///////////////////////////////////////////////////\n')


def print_usage():
    print('Usage:')
    print('    python asynchronous_grab_opencv.py [camera_id]')
    print('    python asynchronous_grab_opencv.py [/h] [-h]')
    print()
    print('Parameters:')
    print('    camera_id   ID of the camera to use (using first camera if not specified)')
    print()


def abort(reason: str, return_code: int = 1, usage: bool = False):
    print(reason + '\n')

    if usage:
        print_usage()

    sys.exit(return_code)


def parse_args() -> Optional[str]:
    args = sys.argv[1:]
    argc = len(args)

    for arg in args:
        if arg in ('/h', '-h'):
            print_usage()
            sys.exit(0)

    if argc > 1:
        abort(reason="Invalid number of arguments. Abort.", return_code=2, usage=True)

    return None if argc == 0 else args[0]


def get_camera(camera_id: Optional[str]) -> Camera:
    with VmbSystem.get_instance() as vmb:
        if camera_id:
            try:
                return vmb.get_camera_by_id(camera_id)

            except VmbCameraError:
                abort('Failed to access Camera \'{}\'. Abort.'.format(camera_id))

        else:
            cams = vmb.get_all_cameras()
            if not cams:
                abort('No Cameras accessible. Abort.')

            return cams[0]


def setup_camera(cam: Camera, exposure, gain):

    with cam:
        # Enable auto exposure time setting if camera supports it
        try:
            # Will also set exposure to 20000us i.e. 20 milliseconds
            print("Default Exposure Time: {0}\n".format(cam.ExposureTime))
            cam.ExposureTime.set(exposure.get())
            print(f"Exposure changed to: {(cam.ExposureTime.get() / 1000):.2f} ms\n\n")

            print("Default Gain: {0}\n".format(cam.Gain))
            cam.Gain.set(gain.get())
            print(f"Gain changed to: {(cam.Gain.get()):.0f}")

        except (AttributeError, VmbFeatureError):
            print("error")
            pass

        # Enable white balancing if camera supports it
        try:
            wba = cam.BalanceWhiteAuto
            print(f"White balance is set to: {(wba.get())}")
            wba.set(1)
            print(f"Changed white balance to: {(wba.get())}")

        except (AttributeError, VmbFeatureError):
            pass


def divergence(ym,distance, width1, width2):
    arr_len = int(((width2 - width1) / 2))
    #left_side = ym[0:arr_len]
    #right_side = ym[arr_len: ]

    max_e2 = np.max(ym) * (1 / np.exp(2))

    for i in range(0, width2 - width1):
        if ym[i] >= max_e2:
            dummy_left = i + 1
            break
        else:
            continue
    for j in range(width2 - width1 - 1, 0, -1):
        if ym[j] >= max_e2:
            dummy_right = j + 1
            break
        else:
            continue

    horizontal_scene_size = (distance/1000)*1134  # mm
    vertical_scene_size = (distance/1000)*851  # mm

    horiz_pix_size =  horizontal_scene_size / 2592
    verti_pix_size = vertical_scene_size /1944

    laser_width = (dummy_right - dummy_left) * horiz_pix_size
    laser_height = (dummy_right - dummy_left) * verti_pix_size

    out = 2 * np.arctan2(laser_width / 2, distance) * 180 / np.pi
    return out

def setup_pixel_format(cam: Camera):
    # Query available pixel formats. Prefer color formats over monochrome formats
    cam_formats = cam.get_pixel_formats()
    cam_color_formats = intersect_pixel_formats(cam_formats, COLOR_PIXEL_FORMATS)
    convertible_color_formats = tuple(f for f in cam_color_formats
                                      if opencv_display_format in f.get_convertible_formats())

    cam_mono_formats = intersect_pixel_formats(cam_formats, MONO_PIXEL_FORMATS)
    convertible_mono_formats = tuple(f for f in cam_mono_formats
                                     if opencv_display_format in f.get_convertible_formats())

    # if OpenCV compatible color format is supported directly, use that
    if opencv_display_format in cam_formats:
        cam.set_pixel_format(opencv_display_format)

    # else if existing color format can be converted to OpenCV format do that
    elif convertible_color_formats:
        cam.set_pixel_format(convertible_color_formats[0])

    # fall back to a mono format that can be converted
    elif convertible_mono_formats:
        cam.set_pixel_format(convertible_mono_formats[0])

    else:
        abort('Camera does not support an OpenCV compatible format. Abort.')


class Handler:
    def __init__(self):
        self.display_queue = Queue(10)

    def get_image(self):
        return self.display_queue.get(True)

    def __call__(self, cam: Camera, stream: Stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            #print('{} acquired {}'.format(cam, frame), flush=True)

            # Convert frame if it is not already the correct format
            if frame.get_pixel_format() == opencv_display_format:
                display = frame
            else:
                # This creates a copy of the frame. The original `frame` object can be requeued
                # safely while `display` is used
                display = frame.convert_pixel_format(opencv_display_format)

            self.display_queue.put(display.as_opencv_image(), True)

        cam.queue_frame(frame)

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        x = int(x * (2592/1280))
        y = int(y * (1944/720))
        print(f'({x}, {y})')


def div_plot(exposure, gain, dist, width1, width2, height1, height2):
    print_preamble()
    cam_id = parse_args()

    with VmbSystem.get_instance():
        with get_camera(cam_id) as cam:
            # setup general camera settings and the pixel format in which frames are recorded
            setup_camera(cam, exposure, gain)
            setup_pixel_format(cam)
            handler = Handler()
            distance = dist.get()
            print("distance is: ", distance)

            try:
                # Start Streaming with a custom a buffer of 10 Frames (defaults to 5)
                cam.start_streaming(handler=handler, buffer_count=10)

                msg = 'Stream from \'{}\'. Press <Enter> to stop stream.'

                ENTER_KEY_CODE = 13
                while True:
                    key = cv2.waitKey(1)
                    if key == ENTER_KEY_CODE:
                        cv2.destroyWindow(msg.format(cam.get_name()))
                        plt.close("all")
                        break
                    cv2.namedWindow(msg.format(cam.get_name()))
                    cv2.setMouseCallback(msg.format(cam.get_name()), click_event)
                    display = handler.get_image()
                    #print(display.shape)


                    #height = image.shape[0]
                    #width = image.shape[1]

                    rng_select = np.mean(display[height1:height2, width1:width2], axis=0)

                    def func(x, a, x0, sigma):
                        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

                    plt.plot(np.arange(width1, width2, 1), rng_select, c='k', label='Function') #Pixel değer plotu

                    try:
                        popt, pcov = curve_fit(func, np.arange(0, width2 - width1, 1), rng_select[:,0])
                        ym = func(np.arange(0, width2 - width1, 1), popt[0], popt[1], popt[2])
                        plt.plot(np.arange(width1, width2, 1), ym, c='r', label='Best fit')
                        plt.title("Divergence Angle: {0}\u00b0".format(str(np.round(divergence(ym, distance, width1, width2))), 3))

                    except:
                        pass

                    #hist = cv2.calcHist([display], [0], None, [256], [0, 256])
                    #plt.hist(display.ravel(), 256, [0, 256]) #Histogram plotu

                    plt.xticks(np.arange(0, width2 - width1, 200))
                    plt.draw()
                    plt.pause(0.05)
                    plt.clf()
                    plt.grid()
                    cv2.rectangle(display, (width1, height2), (width2, height1), (255, 127, 0), thickness=5, lineType=cv2.LINE_8) #ROI
                    display = cv2.resize(display, (1280, 720))

                    cv2.imshow(msg.format(cam.get_name()), display)



            finally:
                cam.stop_streaming()

def histogram(exposure, gain,  width1, width2, height1, height2):
    print_preamble()
    cam_id = parse_args()

    with VmbSystem.get_instance():
        with get_camera(cam_id) as cam:
            # setup general camera settings and the pixel format in which frames are recorded
            setup_camera(cam, exposure, gain)
            setup_pixel_format(cam)
            handler = Handler()

            try:
                # Start Streaming with a custom a buffer of 10 Frames (defaults to 5)
                cam.start_streaming(handler=handler, buffer_count=10)

                msg = 'Stream from \'{}\'. Press <Enter> to stop stream.'

                import cv2
                ENTER_KEY_CODE = 13
                while True:
                    key = cv2.waitKey(1)
                    if key == ENTER_KEY_CODE:
                        cv2.destroyWindow(msg.format(cam.get_name()))
                        plt.close("all")
                        break

                    cv2.namedWindow(msg.format(cam.get_name()))
                    cv2.setMouseCallback(msg.format(cam.get_name()), click_event)
                    display = handler.get_image()

                    image = display

                    #height = image.shape[0]
                    #width = image.shape[1]


                    #display = display[height1:height2,width1 : width2]

                    hist = cv2.calcHist([display], [0], None, [256], [0, 256])
                    plt.hist(display.ravel(), 256, [0, 256]) #Histogram plotu
                    plt.title("Histogram")
                    plt.draw()
                    plt.pause(0.05)
                    plt.clf()
                    plt.grid()
                    cv2.rectangle(display, (width1, height2), (width2, height1), (255, 127, 0), thickness=5,lineType=cv2.LINE_8)  # ROI
                    display = cv2.resize(display, (1280, 720))
                    cv2.imshow(msg.format(cam.get_name()), display)


            finally:
                cam.stop_streaming()

#if __name__ == '__main__':
#    main()


root = Tk()
root.geometry("640x480")

# Pencere başlığı
root.title("Divergence Hesaplama")

exp_label = Label(root, text="Exposure:").place(relx = 0.012, rely = 0.04)
exp_var = DoubleVar()
exp_var.set(100000)
exp = Scale(root, orient=HORIZONTAL, length=500, from_ = 12.957, to = 849053.826 ,variable = exp_var)
exp.pack()

gain_label = Label(root, text="Gain:").place(relx = 0.05, rely = 0.13)
gain_var = DoubleVar()
gai = Scale(root, orient=HORIZONTAL,length=500, from_ = 0.0, to = 24.185 ,variable = gain_var)
gai.pack()

dist_label = Label(root, text="Distance (mm):").place(relx = 0.01, rely = 0.2)
dist_var = DoubleVar()
dist_var.set(2500)
dist = Entry(root,width = 25, textvariable = dist_var, font=('calibre',10,'normal'))
dist.place(relx = 0.16, rely = 0.2)

height1_label = Label(root, text="H1:").place(relx = 0.11, rely = 0.3)
height1_var = IntVar()
height1_e = Entry(root,width = 25, textvariable = height1_var, font=('calibre',10,'normal'))
height1_e.place(relx = 0.16, rely = 0.3)

height2_label = Label(root, text="H2:").place(relx = 0.11, rely = 0.35)
height2_var = IntVar()
height2_var.set(1944)
height2_e = Entry(root,width = 25, textvariable = height2_var, font=('calibre',10,'normal'))
height2_e.place(relx = 0.16, rely = 0.35)

width1_label = Label(root, text="W1:").place(relx = 0.5, rely = 0.3)
width1_var = IntVar()
width1_e = Entry(root,width = 25, textvariable = width1_var, font=('calibre',10,'normal'))
width1_e.place(relx = 0.55, rely = 0.3)

width2_label = Label(root, text="W2:").place(relx = 0.5, rely = 0.35)
width2_var = IntVar()
width2_var.set(2592)
width2_e = Entry(root,width = 25, textvariable = width2_var, font=('calibre',10,'normal'))
width2_e.place(relx = 0.55, rely = 0.35)

width1_var = 0
width2_var = 2592
height1_var = 750
height2_var = 770


exp_but = Button(root, text="Histogram Aç", command=lambda: histogram(exp_var, gain_var, width1_var, width2_var, height1_var, height2_var))
exp_but.place(relx = 0.13, rely = 0.5)


gain_but = Button(root, text="Görüntü Al", command= lambda: div_plot(exp_var, gain_var, dist_var, width1_var, width2_var, height1_var, height2_var))
gain_but.place(relx = 0.13, rely = 0.6)



root.mainloop()

