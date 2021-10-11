import OpenGL.GL as gl
import pangolin
import numpy as np
from time import sleep


def main():
    pangolin.CreateWindowAndBind('Main', 640, 480)
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Define Projection and initial ModelView matrix
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
        pangolin.ModelViewLookAt(-20, -20, -20, 0, 0, 0, pangolin.AxisDirection.AxisY))
    handler = pangolin.Handler3D(scam)

    # Create Interactive View in window
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
    dcam.SetHandler(handler)
    
    T0s = np.fromfile('/home/ronnypetson/repos/cpp_epivo/est.pose', sep=' ')
    T0s = np.reshape(T0s, (-1, 4, 4))
    Ts = np.fromfile('/home/ronnypetson/repos/cpp_epivo/gt.pose', sep=' ')
    Ts = np.reshape(Ts, (-1, 4, 4))
    i = 0
    #sleep(5)
    #pangolin.DisplayBase().RecordOnRender("ffmpeg:[fps=10,bps=8388608,unique_filename]//screencap.avi")
    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        dcam.Activate(scam)
        
        # Draw Point Cloud Nx3
        #points = np.random.random((10000, 3)) * 3 - 4
        #gl.glPointSize(1)
        #gl.glColor3f(1.0, 0.0, 0.0)
        #pangolin.DrawPoints(points)

        # Draw Point Cloud
        #points = np.random.random((10000, 3))
        #colors = np.zeros((len(points), 3))
        #colors[:, 1] = 1 -points[:, 0]
        #colors[:, 2] = 1 - points[:, 1]
        #colors[:, 0] = 1 - points[:, 2]
        #points = points * 3 + 1
        #gl.glPointSize(1)
        #pangolin.DrawPoints(points, colors)

        # j = max(0, i - 10)
        # scam = pangolin.OpenGlRenderState(
        #     pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 200),
        #     pangolin.ModelViewLookAt(Ts[j, 0, 3],
        #                              Ts[j, 1, 3],
        #                              Ts[j, 2, 3],
        #                              0,
        #                              0,
        #                              1,
        #                              pangolin.AxisDirection.AxisNegY))

        # Draw camera
        gl.glLineWidth(2)
        gl.glColor3f(0.0, 1.0, 0.0)
        for pose in Ts[:i + 1]:
            pangolin.DrawCamera(pose, 0.5, 0.75, 0.8)

        gl.glColor3f(1.0, 0.0, 0.0)
        for pose in T0s[:i + 1]:
            pangolin.DrawCamera(pose, 0.5, 0.75, 0.8)

        # Draw line
        gl.glLineWidth(2)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawLine(Ts[:i + 1, :3, 3])

        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawLine(T0s[:i + 1, :3, 3])

        i = min(i + 1, len(Ts) - 1)

        # Draw boxes
        #poses = [np.identity(4) for i in range(10)]
        #for pose in poses:
        #    pose[:3, 3] = np.random.randn(3) + np.array([5,-3,0])
        #sizes = np.random.random((len(poses), 3))
        #gl.glLineWidth(1)
        #gl.glColor3f(1.0, 0.0, 1.0)
        #pangolin.DrawBoxes(poses, sizes)

        pangolin.FinishFrame()
        sleep(0.1)


if __name__ == '__main__':
    main()

