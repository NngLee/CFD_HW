import math
import matplotlib.pyplot as plt
import numpy as np
EPS = 1e-5
MAX_ITER = 10000

def InitVelocity(u, v, xsize, ysize):
    for i in range(xsize):
        u[i][0] = 0.0
        u[i][ysize-1] = math.sin(math.pi * i / xsize) * math.sin(math.pi * i / xsize)
        v[i][0] = 0.0
        v[i][ysize-1] = 0.0
    
    for j in range(ysize):
        u[0][j] = 0.0
        u[xsize-1][j] = 0.0
        v[0][j] = 0.0
        v[xsize-1][j] = 0.0

def InitVorticity(u, v, omega, xsize, ysize, h):
    for i in range(1, xsize-1):
        for j in range(1, ysize-1):
            dv = v[i+1][j] - v[i-1][j]
            du = u[i][j+1] - u[i][j-1]
            omega[i][j] = 0.5 * (dv - du) / h

def ApplyBoundCondition(psi, omega, u, v, xsize, ysize, h):
    # Top and bottom boundaries
    for i in range(xsize):
        omega[i][0] = 2 * (psi[i][0] - psi[i][1]) / (h*h) + 2 * u[i][0] / h
        omega[i][ysize-1] = 2 * (psi[i][ysize-1] - psi[i][ysize-2]) / (h*h) - 2 * u[i][ysize-1] / h
    
    # Left and right boundaries
    for j in range(ysize):
        omega[0][j] = 2 * (psi[0][j] - psi[1][j]) / (h*h) - 2 * v[0][j] / h
        omega[xsize-1][j] = 2 * (psi[xsize-1][j] - psi[xsize-2][j]) / (h*h) + 2 * v[xsize-1][j] / h

def UpdateVorticity(omega, psi, h, nu, dt, xsize, ysize):
    # Create a temporary copy of omega
    temp = [[omega[i][j] for j in range(ysize)] for i in range(xsize)]
    
    for i in range(1, xsize-1):
        for j in range(1, ysize-1):
            laplacian = (omega[i+1][j] + omega[i-1][j] + omega[i][j+1] + omega[i][j-1] - 4*omega[i][j]) / (h*h)
            convection = (psi[i+1][j] - psi[i-1][j]) * (omega[i][j+1] - omega[i][j-1]) / (4*h*h) \
                         - (psi[i][j+1] - psi[i][j-1]) * (omega[i+1][j] - omega[i-1][j]) / (4*h*h)
            temp[i][j] = omega[i][j] + dt * (nu * laplacian + convection)
    
    # Update omega from temp
    for i in range(1, xsize-1):
        for j in range(1, ysize-1):
            omega[i][j] = temp[i][j]

def SOR(f, omega, xsize, ysize, h, relax_factor):
    max_diff = 0.0
    iter_count = 0
    
    while iter_count < MAX_ITER:
        max_diff = 0.0
        for i in range(1, xsize-1):
            for j in range(1, ysize-1):
                old_value = f[i][j]
                new_value = (1 - relax_factor) * old_value + relax_factor * 0.25 * \
                            (f[i+1][j] + f[i][j+1] + f[i-1][j] + f[i][j-1] + h*h * omega[i][j])
                f[i][j] = new_value
                diff = abs(new_value - old_value)
                if diff > max_diff:
                    max_diff = diff
        
        iter_count += 1
        print(iter_count)
        if max_diff <= EPS:
            break
    
    return iter_count < MAX_ITER and iter_count > 1
def plot_contour(x, y, data, name=None):
    plt.figure(figsize=(10, 8))
    plt.contourf(x, y, data, levels=50, cmap="jet")
    plt.colorbar(label=name, orientation='vertical')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(name, pad=15)
    plt.savefig('./' + name + '.png')
    plt.close()

def plot_stream_line(x, y, u, v):

    X, Y = np.meshgrid(x, y)
    # 计算速度大小
    speed = np.sqrt(u**2 + v**2)

    plt.figure(figsize=(10, 8))
    strm = plt.streamplot(X, Y, u, v, color=speed, cmap="jet", density=1.5)
    plt.colorbar(strm.lines, label='Velocity', orientation='vertical')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Streamlines', pad=15)
    plt.savefig('./StreamlinePlot.png')
    plt.close()

def plot_midline_velocity(x, y, u, v):
    plt.figure(figsize=(10, 8))
    midline_velocity = v[int(len(v)/2), :]
    plt.plot(x, midline_velocity, label='Midline Velocity', color='blue')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('v', fontsize=12)
    plt.title('MidlineVelocity', pad=15)
    plt.legend()
    plt.savefig('./MidlineVelocityV.png')
    plt.close()

    plt.figure(figsize=(10, 8))
    midline_velocity = u[:, int(len(u)/2)]
    plt.plot(y, midline_velocity, label='MidlineVelocity', color='blue')
    plt.xlabel('y', fontsize=12)
    plt.ylabel('u', fontsize=12)
    plt.title('MidlineVelocity', pad=15)
    plt.legend()
    plt.savefig('./MidlineVelocityU.png')
    plt.close()

def locate_vortices(data, h):
    for i in range(1, len(data) - 1):
        for j in range(1, len(data[0]) - 1):
            if (data[i][j] < data[i-1][j] and
                data[i][j] < data[i+1][j] and
                data[i][j] < data[i][j-1] and
                data[i][j] < data[i][j+1]):
                print(f"Vortex located at ({j * h}, {i * h}) with stream function value {data[i][j]}")



if __name__ == "__main__":
    h = float(input("Please enter the grid spacing (h): "))
    nu = 0.001
    relax_factor = 1.5
    dt = 0.01
    xsize = int(1.0 / h)
    ysize = int(1.0 / h)
    x = np.linspace(0, 1, int(1/h))
    y = np.linspace(0, 1, int(1/h))
    # Initialize 2D arrays
    u = np.zeros((xsize, ysize))
    v = np.zeros((xsize, ysize))
    omega = np.zeros((xsize, ysize))
    psi = np.zeros((xsize, ysize))
    
    # Initialize velocity and vorticity
    InitVelocity(u, v, xsize, ysize)
    InitVorticity(u, v, omega, xsize, ysize, h)
    
    # Main simulation loop
    is_converge = SOR(psi, omega, xsize, ysize, h, relax_factor)
    while is_converge:
        ApplyBoundCondition(psi, omega, u, v, xsize, ysize, h)
        UpdateVorticity(omega, psi, h, nu, dt, xsize, ysize)
        is_converge = SOR(psi, omega, xsize, ysize, h, relax_factor)
    u = np.gradient (psi, h, axis=0)
    v = -np.gradient(psi,h, axis=1)
    plot_contour(x, y, psi, name="StreamFunction")
    plot_stream_line(x, y, u, v)
    plot_midline_velocity(x, y, u, v)
    locate_vortices(psi, h)
    plot_contour(x, y, omega, name="Vorticity")
    print("All figures saved in ./CFD_HW/HW5/")
    

