import matplotlib.pyplot as plt

def plot(points,points2,x,y):

    plt.scatter([point[0] for point in points], [point[1] for point in points], color='red', label='measurement positions')
    for i in range(len(points) - 1):
        plt.plot([points[i][0], points[i + 1][0]], [points[i][1], points[i + 1][1]], color='red')

    # 添加单独的坐标 x 和 y，黑色
    plt.scatter([point[0] for point in points2], [point[1] for point in points2], color='black', label='filtered positions')
    for i in range(len(points2) - 1):
        plt.plot([points2[i][0], points2[i + 1][0]], [points2[i][1], points2[i + 1][1]], color='black')

    # 添加单独的坐标 x 和 y，绿色
    plt.scatter(x, y, color="green", label='predicted position')


    # 添加标题和图例
    plt.title('Measurement, filtered positions and predicted position')
    plt.legend()

    # 设置坐标轴标签
    plt.xlabel('X')
    plt.ylabel('Y')

    # 显示图形
    plt.grid(True)
    plt.show()


plot(approx3, X_posterior_list_1, x,y)

