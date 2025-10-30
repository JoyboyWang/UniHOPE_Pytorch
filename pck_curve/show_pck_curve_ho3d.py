import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas

colors = [
    [31, 119, 180, 180],   # Blue
    [255, 127, 14, 180],   # Orange
    [44, 160, 44, 180],    # Green
    [180, 39, 40, 180],    # Red
    [148, 103, 189, 180],  # Purple
    [140, 86, 75, 180],    # Brown
    [227, 119, 194, 180],  # Pink
    [174, 199, 232, 180],  # Light Blue
    [188, 189, 34, 180],   # Olive
    [23, 190, 207, 180],   # Teal
    [255, 187, 120, 180],  # Light Orange
    [255, 0, 0, 180]
]


# Function to slightly adjust the colors for better differentiation
def adjust_colors(colors, factor=20):
    new_colors = []
    for color in colors:
        r, g, b, a = color
        r = np.clip(r + np.random.randint(-factor, factor + 1), 0, 255)
        g = np.clip(g + np.random.randint(-factor, factor + 1), 0, 255)
        b = np.clip(b + np.random.randint(-factor, factor + 1), 0, 255)
        new_colors.append([r, g, b, a])
    return new_colors

def show_ho3d_pa_mesh_pck_curve():
    data = pandas.read_excel("ho3d.before_pa_joint_pck_curve.xlsx")
    data = np.array(data)
    data = np.array_split(data, 12, axis=0)
    mpl.rcParams["font.sans-serif"] = "DejaVu Sans"
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.size"] = 17
    mpl.rcParams["figure.figsize"] = (8, 8)
    # data is of size M x N, where M is number of methods, N is the values, evenly distributed from [0,1]

    # before PA joints
    legends = [
        "HandOccNet (CVPR2022): AUC=48.0",
        "MobRecon (CVPR2022): AUC=48.1",
        "H2ONet (CVPR2023): AUC=47.1",
        "SimpleHand (CVPR2024): AUC=47.6",
        "Liu et al. (CVPR2021): AUC=47.6",
        "Keypoint Trans. (CVPR2022): AUC=34.5",
        "HFL-Net (CVPR2023): AUC=50.3",
        "H2ONet$\\dagger$ + HFL-Net$\\dagger$: AUC=47.7",
        "H2ONet$\u2021$ + HFL-Net$\u2021$: AUC=50.4",
        "HandOccNet$\\dagger$ + HFL-Net$\\dagger$: AUC=47.9",
        "HandOccNet$\u2021$ + HFL-Net$\u2021$: AUC=50.4",
        "Ours: AUC=52.3"
    ]
    
    fig = plt.figure(1)
    fig.clear()

    plt.subplot(111)
    title = "HO3D Joint AUC"
    xlabel = "Error thresholds (mm)"
    ylabel = "Joint 3D PCK"
    # fig.show()
    plt.grid(linestyle="solid")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(np.arange(10., 50.01, step=10))
    # plt.xticks(np.arange(0., 50.01, step=10))
    

    ax = fig.gca()
    ax.set_xlim(10, 50)
    ax.set_ylim(0.1, 0.95)
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels = ["10", "20", "30", "40", "50"]

    ax.set_xticklabels(labels)

    markers = ["o", "D", "^"]

    for idx, xy in enumerate(data):
        color = [x / 255.0 for x in colors[idx % 12]]
        color_fill = [*color[0:3], color[3] * 0.5]
        
        ax.plot(xy[0, 10:] * 10, xy[1, 10:], linestyle="--", color=color, lw=2.7, mew=1.6, ms=8.5, mec=color, mfc=color_fill, label=legends[idx])
        

    ax.legend(loc="best", shadow=False, fontsize="small", fancybox=True, framealpha=0.75, ncol=1)

    fig.canvas.draw()
    fig.savefig("ho3d.before_pa_joint_pck_curve.svg", dpi=1200, bbox_inches='tight')


if __name__ == '__main__':
    show_ho3d_pa_mesh_pck_curve()