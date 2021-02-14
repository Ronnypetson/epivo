import OpenGL.GL as gl
import pangolin
import numpy as np
from time import sleep


def main():
    h, w = 480, 752 # 640, 480
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

    ## KITTI
    #gTs = np.fromfile('/home/ronnypetson/dataset/poses/00.txt', sep=' ')
    gTs = np.fromfile('/home/ronnypetson/repos/cpp_epivo/kitti.GT', sep=' ')
    gTs = np.reshape(gTs, (-1, 4, 4))
    Ts = np.fromfile('/home/ronnypetson/repos/cpp_epivo/kitti.T', sep=' ')
    dummy = np.array([0.0, 0.0, 0.0, 1.0]).reshape(1, 4)

    ## Euroc
    #Ts = np.fromfile('/home/ronnypetson/repos/cpp_epivo/euroc.T', sep=' ')
    Ts = np.reshape(Ts, (-1, 4, 4))
    T_DC = np.array([[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                     [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                     [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                     [0.0, 0.0, 0.0, 1.0]])
    T_DC_ = np.linalg.inv(T_DC)

    R_z = np.array([[0.0, -1.0, 0.0, 0.0],
                    [1.0,  0.0, 0.0, 0.0],
                    [0.0,  0.0, 1.0, 0.0],
                    [0.0,  0.0, 0.0, 1.0]])
    i = 0
    #sleep(5)
    #pangolin.DisplayBase().RecordOnRender("ffmpeg:[fps=10,bps=8388608,unique_filename]//kitti_00.avi")
    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        dcam.Activate(scam)
        
        #T_wc = scam.GetModelViewMatrix()

        ## KITTI
        #gTs_ = np.linalg.inv(np.concatenate([gTs[i], dummy], axis=0))
        gTs_ = np.linalg.inv(gTs[i])
        gTs_[1] *= -1.0
        gTs_[2] *= -1.0
        gTs_[2, 3] -= 50.0

        ## Euroc
        Ts_ = Ts[i] # Euroc

        ## KITTI
        # Forward lookin: negate y and z. Backward: negate x and y
        #Ts_[0] *= -1.0
        #Ts_[1] *= -1.0
        ##Ts_[2] *= -1.0


        ## Euroc
        #Ts_ = R_z * Ts_
        Ts_ = np.linalg.inv(Ts_)
        Ts_[1] *= -1.0
        Ts_[2] *= -1.0
        #Ts_[2, 3] -= 10.0

        scam.SetModelViewMatrix(pangolin.OpenGlMatrix(Ts_))
        dcam.Render()

        # Draw camera
        gl.glLineWidth(2)
        gl.glColor3f(1.0, 0.0, 0.0)
        for pose in Ts[:i + 1:10]:
            pangolin.DrawCamera(pose, 0.5/2, 0.75/2, 0.8/2)

        gl.glColor3f(0.0, 1.0, 0.0)
        for pose in gTs[:i + 1:10]:
            pangolin.DrawCamera(pose, 0.5/2, 0.75/2, 0.8/2)

        # Draw Point Cloud Nx3
        gl.glPointSize(2)
        gl.glColor3f(0.0, 0.0, 1.0)
        pangolin.DrawPoints(X[:L[i + 1]])

        if i == len(L) - 2:
            break
        i = min(i + 1, len(L) - 2)

        pangolin.FinishFrame()
        #sleep(0.1)


if __name__ == '__main__':
    main()

