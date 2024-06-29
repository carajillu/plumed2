import matplotlib.pyplot as plt
import numpy as np

def S_on(m):
    if m < 0:
        return 0
    elif 0 <= m <= 1:
        return 3*m**4 - 2*m**6
    else:
        return 1
   
def S_off(m):
    if m < 0:
        return 1
    elif 0 <= m <= 1:
        return 3*(m-1)**4 - 2*(m-1)**6
    else:
        return 0
    

if __name__=="__main__":

    m = np.linspace(-0.2, 1.2, 1000)

    S_on_values = [S_on(m_) for m_ in m]
    S_off_values = [S_off(m_) for m_ in m]

    fig, ax1 = plt.subplots(figsize=(6.4, 4.8))

    ax1.set_xlabel(r"$\nu$", fontweight='bold')
    ax1.set_ylabel(r"$S$", fontweight='bold')

    ax1.axvline(x=0, color="black", linestyle="dotted", linewidth=2)
    ax1.text(-0.2, 0.9, r"$\nu_{0}$", verticalalignment='top', horizontalalignment='left', color='black',fontweight="bold")

    ax1.axvline(x=1, color="black", linestyle="dotted", linewidth=2)
    ax1.text(1.05, 0.9, r"$\nu_{0}+\Delta\nu$", verticalalignment='top', horizontalalignment='left', color='black',fontweight="bold")

    ax1.text(-0.15, 0.75, "CUTOFF\nZONE", verticalalignment='top', horizontalalignment='center', color='black',fontweight="bold")
    ax1.text(0.50, 0.75, "CALCULATION\nZONE", verticalalignment='top', horizontalalignment='center', color='black',fontweight="bold")
    ax1.text(1.15, 0.75, "CUTOFF\nZONE", verticalalignment='top', horizontalalignment='center', color='black',fontweight="bold")

    plt.plot(m, S_on_values, label=r"$S^{on}(\nu,\nu_{0},\Delta\nu)$",linewidth=5,color='blue')
    plt.plot(m, S_off_values, label=r"$S^{off}(\nu,\nu_{0},\Delta\nu)$",linewidth=5,color='red')
    plt.legend()
    plt.show()