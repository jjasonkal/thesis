import matplotlib.pyplot as plt
import math

def lr_poly_old(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def lr_poly(base_lr, iter, max_iter, power, warm_iter):
    if iter < warm_iter:
        return base_lr * iter / warm_iter
    return base_lr * ((1 - float(iter-warm_iter) / (max_iter-warm_iter)) ** (power))


def lr_cos(base_lr, iter, max_iter, power, warm_iter):
    if iter < warm_iter:
        return base_lr * iter / warm_iter
    return base_lr * 0.5 * (1+ math.cos(math.pi*float(iter-warm_iter) / (max_iter-warm_iter)))
    # return base_lr * 0.5 * (1+ math.cos(math.pi*float(iter) / (max_iter)))
    # return base_lr * ((1 - float(iter-warm_iter) / (max_iter-warm_iter)) ** (power)) * 0.5 * (1+ math.cos(math.pi*float(iter-warm_iter) / (max_iter-warm_iter)))


max_iterations = 250000
learning_rate = 2.5e-4
lr_schedule_power = 0.9
warm_iter = 10000
lr_old = [lr_poly_old(learning_rate, x, max_iterations, lr_schedule_power) for x in range(1, max_iterations)]
lr = [lr_poly(learning_rate, x, max_iterations, lr_schedule_power, warm_iter) for x in range(1, max_iterations)]
lr_cosine = [lr_cos(learning_rate, x, max_iterations, lr_schedule_power, warm_iter) for x in range(1, max_iterations)]
plt.plot(range(1, max_iterations), lr_old)
plt.plot(range(1, max_iterations), lr)
plt.plot(range(1, max_iterations), lr_cosine)
plt.xlabel("Επαναλήψεις")
plt.ylabel("Ρυθμός Μάθησης")
plt.show()
plt.savefig('lr_cos_norm.png')
print('asas')