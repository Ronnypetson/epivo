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
        pangolin.ModelViewLookAt(0, 0, 0, 0, 0, 1, pangolin.AxisDirection.AxisNegY))
    handler = pangolin.Handler3D(scam)

    # Create Interactive View in window
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
    dcam.SetHandler(handler)
    
    T_wc = scam.GetModelViewMatrix()

    X = np.fromfile('/home/ronnypetson/repos/cpp_epivo/pts.cld', sep=' ')
    X = np.reshape(X, (-1, 3))
    L = np.fromfile('/home/ronnypetson/repos/cpp_epivo/lims', sep=' ')
    L = [int(l) for l in np.reshape(L, (-1,)).tolist()]
    Ts = np.fromfile('/home/ronnypetson/dataset/poses/00.txt', sep=' ')
    Ts = np.reshape(Ts, (-1, 3, 4))
    dummy = np.array([0.0, 0.0, 0.0, 1.0]).reshape(1, 4)
    R_z = np.eye(3)
    #R_z[0, 0] = -1.0
    #R_z[1, 1] = -1.0
    #R_z = np.linalg.inv(R_z)
    i = 0
    #sleep(5)
    pangolin.DisplayBase().RecordOnRender("ffmpeg:[fps=10,bps=8388608,unique_filename]//pt_cloud_mov.avi")
    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        dcam.Activate(scam)
        
        #T_wc = scam.GetModelViewMatrix()
        Ts_ = np.linalg.inv(np.concatenate([Ts[i], dummy], axis=0))
        Ts_[0] *= -1.0
        Ts_[1] *= -1.0
        #Ts_[:3, :3] = R_z * Ts_[:3, :3]
        #Ts_ = np.concatenate([Ts[i], dummy], axis=0)
        #T_wc_ = T_wc * pangolin.OpenGlMatrix(Ts_)
        scam.SetModelViewMatrix(pangolin.OpenGlMatrix(Ts_))
        dcam.Render()

        # Draw Point Cloud Nx3
        #points = np.random.random((10000, 3)) * 3 - 4
        gl.glPointSize(2)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawPoints(X[:L[i + 1]])

        if i == len(L) - 2:
            break
        i = min(i + 1, len(L) - 2)

        pangolin.FinishFrame()
        sleep(0.1)


if __name__ == '__main__':
    main()

