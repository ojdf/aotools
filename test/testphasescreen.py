from aotools import phasescreen

def test_ftScrn():

    scrn = phasescreen.ft_phase_screen(0.2, 512, 4.2/128, 30., 0.01)

def test_ftShScrn():
    scrn = phasescreen.ft_sh_phase_screen(0.2, 512, 4.2/128, 30., 0.01)
