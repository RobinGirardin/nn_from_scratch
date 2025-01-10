import matplotlib.pyplot as plt


def get_image(img):
    image = img.reshape((28, 28))

    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    plt.show()
