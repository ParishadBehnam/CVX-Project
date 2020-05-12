import matplotlib.pyplot as plt
import numpy as np


def generate_k(training_data_x, training_data_y):
    global n, m
    k_tilda = np.zeros((n, n))
    k = np.zeros((n, n))

    for i in range(training_data_y.shape[0]):
        for j in range(training_data_y.shape[0]):
            k_tilda[i, j] = np.exp(-500 * np.linalg.norm(training_data_x[i] - training_data_x[j]) ** 2) \
                            * training_data_y[i] * training_data_y[j]
            k[i, j] = np.exp(-500 * np.linalg.norm(training_data_x[i] - training_data_x[j]) ** 2)
    return k, k_tilda


def barrier(w, t):
    global k_tilda, A, b
    return -t * np.sum(w) + 0.5 * np.dot(np.dot(w.T, k_tilda), w) - np.sum(np.log(b - np.dot(A, w)))


def grad_barrier(w, t):
    global k_tilda, A, b
    return -t * np.ones_like(w) + t * np.dot(k_tilda, w) + np.sum(A * (1 / (b - np.dot(A, w))), axis=0).reshape(w.shape)


def hessian_barrier(w, t):
    global k_tilda, A, b
    eye = np.vstack([np.eye(w.shape[0]), np.eye(w.shape[0])])
    f = np.diag(np.reshape(1 / (b - np.dot(A, w)) ** 2, -1))
    return t * k_tilda + np.dot(np.dot(eye.T, f), eye)


def train_svm(data_y):
    global t, c, n, mio, alpha, beta, A, b

    scores = []
    p = sum(data_y > 0) / n
    w = np.where(data_y > 0, data_y * c * (1 - p), -data_y * c * p).reshape(n, 1) / 100

    count = 0

    # log barrier method
    while n / t > 1e-8:
        print('run', count, 'm/t', n / t)
        count += 1
        delta_x = -np.dot(np.linalg.inv(hessian_barrier(w, t)), grad_barrier(w, t))
        lambda2 = np.dot(np.dot(grad_barrier(w, t).T, np.linalg.inv(hessian_barrier(w, t))), grad_barrier(w, t))

        # newton's method
        while lambda2[0, 0] > 2e-16:
            # print('lambda', lambda2)

            # step defining
            step = 1
            # Update step until w in dom(f)
            while np.sum(A.dot(w + step * delta_x) < b) < b.shape[0]:
                step *= beta
            while barrier(w + step * delta_x, t) > barrier(w, t) + alpha * step * np.dot(grad_barrier(w, t).T, delta_x):
                step *= beta
            w = w + step * delta_x

            prev_lambda = lambda2
            delta_x = -np.dot(np.linalg.inv(hessian_barrier(w, t)), grad_barrier(w, t))
            lambda2 = np.dot(np.dot(grad_barrier(w, t).T, np.linalg.inv(hessian_barrier(w, t))), grad_barrier(w, t))
            if np.abs(lambda2 - prev_lambda) < 1e-10:
                # print('break')
                break
        scores.append(barrier(w, t)[0, 0])
        t *= mio
    return w, scores


alpha = 0.01
beta = 0.5
c = 0.1
t = 1e4
mio = 8
m = 2
n = 863
A = np.vstack([np.eye(n), -np.eye(n)])
b = np.vstack([c * np.ones((n, 1)), np.zeros((n, 1))])

# load data
training_data_y = np.zeros(n)
training_data_x = np.zeros((n, m))
with open('data.txt') as fp:
    data = fp.readline()
    i = 0
    while data:
        training_data_x[i, 0], training_data_x[i, 1], training_data_y[i] = np.array(data.split()).astype('float')
        i += 1
        data = fp.readline()

k, k_tilda = generate_k(training_data_x, training_data_y)

print('start')
weight, scores = train_svm(training_data_y)

# supporting vectors
svs = np.sum(weight > 1e-5)
print('number of support vectors:\t', svs)

# bias calculating
w_0 = np.sum(
    np.where(weight > 1e-5,
             training_data_y.reshape(n, 1) - np.dot((weight * training_data_y.reshape(n, 1)).T, k).reshape(n, 1),
             np.zeros_like(weight))) / svs

#test
y_hat = np.sign(np.dot(k, weight * training_data_y.reshape(n, 1)) + w_0)

# accuracy
acc = np.sum(training_data_y == y_hat.reshape(n)) / n
print('accuracy', acc)


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(scores)
ax1.set_title('scores through iter')
ax1.set_xlabel('iteration')
ax1.set_ylabel('score')
ax2.scatter(training_data_x[:, 0], training_data_x[:, 1], c=training_data_y, s=1)
ax2.set_title('original data')
ax3.scatter(training_data_x[:, 0], training_data_x[:, 1], c=y_hat.reshape(n), s=1)
ax2.set_title('predicted data')
fig.delaxes(ax4)
fig.tight_layout()
plt.show()


