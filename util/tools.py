from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

def show_img(img_data, text):
    # 입력된 값이 0~1로 정규화된 값이므로 255를 곱함
    _img_data = img_data * 255
    
    # 4d tensor -> 2d
    _img_data = np.array(_img_data[0,0], dtype=np.uint8)

    img_data = Image.fromarray(_img_data)
    draw = ImageDraw.Draw(img_data)

    # 예측결과을 텍스트로 보기 위해 선언
    # draw text in img, center_x, center_y
    cx, cy = _img_data.shape[0] / 2, _img_data.shape[1] / 2
    if text is not None:
        draw.text((cx,cy), text)

    plt.imshow(img_data)
    plt.show()

def result_subplot(epochs, history):
    history = np.array(history)
    loss = history[0][0]
    acc = history[0][1]
    x = np.linspace(0, epochs, epochs)
    out = [loss, acc]
    for i in range(2):
        plt.subplot(1,2,i+1)
        plt.plot(x, out[i])
        plt.title('training' + str(out[i]))
        plt.legend()
        plt.tight_layout()
    plt.show()

def result_plot(history):
    history = np.array(history)
    plt.plot(history[:,0:2])
    plt.text(0,0, "max accuracy" + str(max(history[:,0])))
    plt.legend(['loss', 'accuracy'])
    plt.show()
