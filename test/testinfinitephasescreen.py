from aotools.turbulence import infinitephasescreen, infinitephasescreen_fried

def testInitScreen():

    scrn = infinitephasescreen.PhaseScreen(128, 4./64, 0.2, 50, nCol=4)

def testAddRow_axis0_forward():

    scrn = infinitephasescreen.PhaseScreen(128, 4./64, 0.2, 50, nCol=4)
    scrn.addRow(1, axis=0)

def testAddRow_axis0_backward():

    scrn = infinitephasescreen.PhaseScreen(128, 4./64, 0.2, 50, nCol=4)
    scrn.addRow(-1, axis=0)

def testAddRow_axis1_forward():

    scrn = infinitephasescreen.PhaseScreen(128, 4./64, 0.2, 50, nCol=4)
    scrn.addRow(1, axis=1)

def testAddRow_axis1_backward():

    scrn = infinitephasescreen.PhaseScreen(128, 4./64, 0.2, 50, nCol=4)
    scrn.addRow(-1, axis=0)


def testAddMultipleRows():

    scrn = infinitephasescreen.PhaseScreen(128, 4./64, 0.2, 50, nCol=4)
    scrn.addRow(10, axis=0)

def testMoveScrn_axis0_forward():

    scrn = infinitephasescreen.PhaseScreen(128, 4./64, 0.2, 50, nCol=4)
    scrn.moveScrn((0.3, 0))

def testMoveScrn_axis0_backward():

    scrn = infinitephasescreen.PhaseScreen(128, 4./64, 0.2, 50, nCol=4)
    scrn.moveScrn((-0.3, 0))

def testMoveScrn_axis1_forward():

    scrn = infinitephasescreen.PhaseScreen(128, 4./64, 0.2, 50, nCol=4)
    scrn.moveScrn((0, 0.3))

def testMoveScrn_axis1_backward():

    scrn = infinitephasescreen.PhaseScreen(128, 4./64, 0.2, 50, nCol=4)
    scrn.moveScrn((0, -0.3))


def testMoveDiagonal1():

    scrn = infinitephasescreen.PhaseScreen(128, 4./64, 0.2, 50, nCol=4)
    scrn.moveScrn((0.3, 0.3))


def testMoveDiagonal2():

    scrn = infinitephasescreen.PhaseScreen(128, 4./64, 0.2, 50, nCol=4)
    scrn.moveScrn((0.3, -0.3))


# Test of Fried screen
def test_fried_init():
    screen = infinitephasescreen_fried.PhaseScreen(64, 8./32, 0.2, 40)

def test_fried_add_row():
    screen = infinitephasescreen_fried.PhaseScreen(64, 8./32, 0.2, 40)

    screen.addRow()

if __name__ == "__main__":

    from matplotlib import pyplot

    screen = infinitephasescreen_fried.PhaseScreen(64, 8./32, 0.2, 40, 2)

    pyplot.ion()
    pyplot.imshow(screen.stencil)

    pyplot.figure()
    pyplot.imshow(screen.scrn)
    pyplot.colorbar()
    for i in range(100):
        screen.addRow()

        pyplot.clf()
        pyplot.imshow(screen.scrn)
        pyplot.colorbar()
        pyplot.draw()
        pyplot.pause(0.01)
