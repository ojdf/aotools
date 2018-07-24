from aotools import karhunenLoeve as KL


def test_make():
    kl, _, _, _ = KL.make_kl(150, 128, ri=0.2, stf='kolmogorov')
