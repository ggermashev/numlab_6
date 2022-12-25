import numpy as np
from matplotlib import pyplot as plt

left_edge = [1.0, 1.0, 1.3333]
right_edge = [1.0, 0.0, 0.9280]


def p(x): return -np.tanh(x)


def q(x): return np.cosh(x) ** 2


def f(x): return (x * np.cosh(x) ** 2 - np.tanh(x)) / 3


start = 0.0
end = 1.0
count = 21
nodes = np.linspace(start, end, count)


def create_matrix(p, q, f, lft_edge, rgt_edge, nodes, accuracy=1):
    h = nodes[1] - nodes[0]
    left_matrix = np.zeros(3 * len(nodes) - 2)
    right_matrix = np.zeros(len(nodes))
    i = 3
    for k in range(1, len(nodes) - 1):
        left_matrix[i - 1] = 1 - p(nodes[k]) * h / 2  # ak
        left_matrix[i] = -2 + h * h * q(nodes[k])  # bk
        left_matrix[i + 1] = 1 + h * p(nodes[k]) / 2  # ck
        right_matrix[k] = h * h * f(nodes[k])  # dk
        i += 3
    if accuracy == 1:
        left_matrix[0] = lft_edge[1] * h - lft_edge[0]  # b0
        left_matrix[1] = lft_edge[0]  # c0
        right_matrix[0] = lft_edge[2] * h  # d0
        left_matrix[-2] = -1 * rgt_edge[0]  # an
        left_matrix[-1] = rgt_edge[0] + h * rgt_edge[1]  # bn
        right_matrix[-1] = rgt_edge[2] * h  # dn
    elif accuracy == 2:
        if lft_edge[0] != 0:
            left_matrix[0] = -2 + 2 * h * lft_edge[1] / lft_edge[0] - p(nodes[0]) * lft_edge[1] * h * h / lft_edge[0] + q(nodes[0]) * h * h
            left_matrix[1] = 2
            right_matrix[0] = h * h * f(nodes[0]) + 2 * h * lft_edge[2] / lft_edge[0] - p(nodes[0]) * lft_edge[2] * h * h / lft_edge[0]
        else:
            left_matrix[0] = lft_edge[1]
            left_matrix[1] = 0
            right_matrix[0] = lft_edge[2]
        if rgt_edge[0] != 0:
            left_matrix[-2] = 2
            left_matrix[-1] = -2 - 2 * h * rgt_edge[1] / rgt_edge[0] - p(nodes[-1]) * h * h * rgt_edge[1] / rgt_edge[0] + q(nodes[-1]) * h * h
            right_matrix[-1] = f(nodes[-1]) * h * h - 2 * h * rgt_edge[2] / rgt_edge[0] - p(nodes[-1]) * h * h * rgt_edge[2] / rgt_edge[0]
        else:
            left_matrix[-2] = 0
            left_matrix[-1] = rgt_edge[1]
            right_matrix[-1] = rgt_edge[2]
    return left_matrix, right_matrix


def solve_eqv(left_matrix, right_matrix):
    a = np.zeros(len(right_matrix))
    b = np.zeros(len(right_matrix))
    ans = np.zeros(len(right_matrix))

    a[0] = right_matrix[0] / left_matrix[0]
    b[0] = -1 * left_matrix[1] / left_matrix[0]
    i = 3
    for k in range(1, len(right_matrix) - 1):
        a[k] = (right_matrix[k] - left_matrix[i - 1] * a[k - 1]) / (left_matrix[i] + left_matrix[i - 1] * b[k - 1])
        b[k] = -1 * left_matrix[i + 1] / (left_matrix[i] + left_matrix[i - 1] * b[k - 1])
        i += 3
    a[-1] = (right_matrix[-1] - left_matrix[i - 1] * a[-2]) / (left_matrix[i] + left_matrix[i - 1] * b[-2])
    b[-1] = 0.0

    ans[-1] = a[-1]
    for i in range(len(ans) - 2, -1, -1):
        ans[i] = a[i] + b[i] * ans[i + 1]

    return ans


def answer(x):
    return np.sin(np.sinh(x)) + x / 3


def get_error(left_matrix, right_matrix, answer):
    return np.max(np.abs(answer - solve_eqv(left_matrix, right_matrix)))



left_m1, right_m1 = create_matrix(p, q, f, left_edge, right_edge, nodes, 1)
left_m2, right_m2 = create_matrix(p, q, f, left_edge, right_edge, nodes, 2)
ans1 = solve_eqv(left_m1, right_m1)
ans2 = solve_eqv(left_m2, right_m2)
ans = [answer(x) for x in nodes]

plt.plot(nodes, ans, label='ans')
plt.plot(nodes, ans1, label='1 accuracy')
plt.plot(nodes, ans2, label='2 accuracy')
plt.legend()
plt.show()


counts = np.arange(100, 0, -5)
errors1 = []
errors2 = []
steps = []
for count in counts:
    nodes = np.linspace(start, end, count)
    steps.append(nodes[1] - nodes[0])
    left_m1, right_m1 = create_matrix(p, q, f, left_edge, right_edge, nodes, 1)
    left_m2, right_m2 = create_matrix(p, q, f, left_edge, right_edge, nodes, 2)
    ans1 = solve_eqv(left_m1, right_m1)
    ans2 = solve_eqv(left_m2, right_m2)
    ans = [answer(x) for x in nodes]
    errors1.append(get_error(left_m1, right_m1, ans))
    errors2.append(get_error(left_m2, right_m2, ans))


y_x = [x for x in steps]
y_2x = [x ** 2 for x in steps]
plt.subplot(1, 2, 1)
plt.loglog(steps, errors1, label='accuracy 1 errors')
plt.loglog(steps, y_x, label='y = x')
plt.legend()
plt.subplot(1, 2, 2)
plt.loglog(steps, errors2, label='accuracy 2 errors')
plt.loglog(steps, y_2x, label='y = 2x')
plt.legend()
plt.show()
