from aotools.phasescreen import infinitephasescreen

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