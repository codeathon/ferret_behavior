import sys, time, cv2, ffmpeg, numpy
import fcntl

from src.utilities.logging_config import get_logger

logger = get_logger(__name__)
videoCapture = cv2.VideoCapture('/home/scholl-lab/recordings/session_2025-06-06/framerate_testing__1/raw_videos/24908831.mp4')
process = (
    ffmpeg
    .input('pipe:', framerate='{}'.format(videoCapture.get(cv2.CAP_PROP_FPS)), format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    .output('ffmpeg_test.mp4', vcodec='h264_nvenc', pix_fmt='yuv420p', **{'b:v': 2000000})
    .overwrite_output()
    .run_async(pipe_stdin=True, quiet=True)
)
fd = process.stdin.fileno()
pipe_size = 200000000
logger.info("pipe size before: %d", fcntl.fcntl(fd, fcntl.F_GETPIPE_SZ))
fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, pipe_size)
logger.info("pipe size after:  %d", fcntl.fcntl(fd, fcntl.F_GETPIPE_SZ))
process.stdin.close()
process.wait()
# lastFrame = False
# frames = 0
# start = time.time()
# while not lastFrame:
#     ret, image = videoCapture.read()
#     if ret:
#         process.stdin.write(
#             image
#             .astype(numpy.uint8)
#             .tobytes()
#         )        
#         frames += 1
#     else:
#         lastFrame = True
# elapsed = time.time() - start
# logger.info("%d frames" % frames)
# logger.info("%4.1f FPS, elapsed time: %4.2f seconds" % (frames / elapsed, elapsed))
videoCapture.release()