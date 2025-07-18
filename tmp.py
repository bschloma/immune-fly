def compute_random_jump_distance_from_uniform(u, μ):
    return u ** (- 1 / μ)


def draw_random_angle(u):
    return 2 * xp.pi * u


def update_virus_grid(grid, μ, u_arr, ic_locs=None, grid_pad=0):
    u1, u2, u3 = u_arr

    # pick random occupied site
    occupied_sites = xp.argwhere(grid == 1)
    if len(occupied_sites) == 0:
        return grid
    y_start, x_start = occupied_sites[int(u1 * len(occupied_sites))]

    # draw random jump distance
    r = compute_random_jump_distance_from_uniform(u2, μ)

    # pick random jump angle
    angle = draw_random_angle(u3)

    # find new site
    # x_land = int(xp.round((x_start + r * xp.cos(angle))))
    # y_land = int(xp.round((y_start + r * xp.sin(angle))))
    # find new site accounting for periodic boundary conditions
    L = grid.shape[0]
    x_land = int(xp.round((x_start + r * xp.cos(angle) - grid_pad))) % (L - 2 * grid_pad) + grid_pad
    y_land = int(xp.round((y_start + r * xp.sin(angle) - grid_pad))) % (L - 2 * grid_pad) + grid_pad

    # check if new site lives in the grid
    # TODO: periodic boundary conditions
    #     if y_land >= grid.shape[0]:
    #         y_land = grid.shape[0] - 1
    #     if x_land >= grid.shape[1]:
    #         x_land = grid.shape[0] - 1
    # periodic boundary conditions
    #     if x_land >= L - grid_pad:
    #         x_land = grid_pad + 1 + np.mod(x_land, L - grid_pad)
    #     if y_land >= L - grid_pad:
    #         y_land = grid_pad + 1 + np.mod(y_land, L - grid_pad)
    #     if x_land <= grid_pad:
    #         x_land = L - grid_pad - 1 - grid_pad + x_land
    #     if y_land <= grid_pad:
    #         y_land = L - grid_pad - 1 - grid_pad + y_land

    # update site
    grid[y_land, x_land] = True

    # optional---check if new site is occupied by an immune cell. if so, the virus is immediately killed there
    if ic_locs is not None:
        this_loc = xp.array([y_land, x_land])
        if any(xp.sum(ic_locs - this_loc, axis=1) == 0):
            grid[y_land, x_land] = False

    return grid


def jump_immune_cell(subgrid, ic_loc, μ, random_number, L, max_std_angle=xp.pi, ic_angle=0, K=0.01):
    """pass a subgrid to use for gradient computation"""
    virus_gradient_y, virus_gradient_x = xp.gradient(subgrid)
    virus_gradient_magnitude = xp.sqrt((virus_gradient_y ** 2 + virus_gradient_x ** 2))
    virus_gradient_angle = xp.arctan2(virus_gradient_y, virus_gradient_x)

    virus_subgrid_size = len(subgrid)

    virus_angle = virus_gradient_angle[virus_subgrid_size // 2 - 1, virus_subgrid_size // 2 - 1]

    local_virus = subgrid[virus_subgrid_size // 2 - 1, virus_subgrid_size // 2 - 1]
    local_virus_gradient = virus_gradient_magnitude[virus_subgrid_size // 2 - 1, virus_subgrid_size // 2 - 1]

    mean_angle = ic_angle * sensing_function(local_virus, local_virus_gradient, K) + (
                1 - sensing_function(local_virus, local_virus_gradient, K)) * virus_angle

    # compute standard deviation of the angle.
    std_angle = max_std_angle * sensing_function(local_virus, local_virus_gradient, K)

    new_angle = xp.random.normal(loc=mean_angle, scale=std_angle)
    jump_distance = compute_random_jump_distance_from_uniform(random_number, μ)

    y, x = ic_loc
    shifted_y = y - L / 2
    shifted_x = x - L / 2
    new_shifted_x = int(xp.around(shifted_x + jump_distance * xp.cos(new_angle)))
    new_shifted_y = int(xp.around(shifted_y + jump_distance * xp.sin(new_angle)))
    new_x = new_shifted_x + L / 2
    new_y = new_shifted_y + L / 2

    return new_x, new_y, new_angle


def screened_poisson_green_kernel_2d(lam, size):
    """Generate discrete 2D Green's function kernel for (∇² - λ^2)G = -δ"""
    N = size // 2
    kernel = np.zeros((size, size))
    for m in range(-N, N + 1):
        for n in range(-N, N + 1):
            r = np.sqrt(m ** 2 + n ** 2)
            if r > 0:
                kernel[m + N, n + N] = k0(lam * r)
            else:
                # Use limiting value of K0 near r = 0 (log divergence, but you can regularize)
                kernel[m + N, n + N] = (-np.log(lam / 2) - 0.5772)
    return kernel / np.sum(kernel)  # Optional: normalize


# def compute_signal_grid(virus_grid, lam):

def run_simulation(L=2 ** 8,
                   n_ics=int(4 ** 2),
                   n_steps=100000,
                   t_eval=None,
                   μ_I=2.5,
                   μ_V=2.5,
                   p_replication=0.01,
                   lam=5,
                   grid_pad=4,
                   kernel_size=7,
                   virus_grid=None,
                   max_std_angle=xp.pi):
    """initialize virus"""
    if virus_grid is None:
        virus_grid = xp.zeros((L, L), dtype='bool')
        virus_grid[int(L / 2), int(L / 2)] = True
    signal_grid = xp.zeros((L, L), dtype='float32')
    kernel = xp.array(screened_poisson_green_kernel_2d(lam, size=kernel_size))

    """initialize immune cells"""
    n_xy = int(xp.sqrt(n_ics))
    ic_x = xp.linspace(grid_pad + 1, L - grid_pad - 1, n_xy)
    ic_y = xp.linspace(grid_pad + 1, L - grid_pad - 1, n_xy)
    ic_locs = xp.zeros((n_ics, 2))
    counter = 0
    for i in range(n_xy):
        for j in range(n_xy):
            ic_locs[counter] = xp.array([int(ic_y[j]), int(ic_x[i])])
            counter += 1
    # ic_locs[0] = xp.array([int(L / 4), int(L / 4) - 30])
    ic_angles = 2 * np.pi * (xp.random.random(size=len(ic_locs)) - 0.5)

    """initialize output"""
    if t_eval is None:
        # just output the final values
        t_eval = np.array([n_steps - 1])
    virus_out = np.zeros((len(t_eval), L, L))
    ic_locs_out = np.zeros((len(t_eval), n_ics, 2))
    t_counter = 0
    if t_eval[0] == 0:
        virus_out[t_counter] = virus_grid.get()
        ic_locs_out[t_counter] = ic_locs.get()
        t_counter += 1

    """initialize random numbers"""
    random_numbers_replication = np.random.random(n_steps)
    random_numbers_virus_jump = xp.random.uniform(size=(n_steps, 3))
    random_numbers_immune_jump = np.random.random(n_steps)

    """run the simulation"""
    for i in tqdm(range(n_steps)):
        # update signal
        signal_grid = convolve(virus_grid.astype('float32'), kernel, mode='wrap')
        # signal_grid += 0.1 * (convolve(signal_grid, laplacian_kernel(), mode='wrap') - 0.01 * signal_grid + virus_grid)

        if random_numbers_replication[i] <= p_replication:
            # update virus
            virus_grid = update_virus_grid(virus_grid, μ_V, random_numbers_virus_jump[i], ic_locs, grid_pad=grid_pad)

        else:
            # update immune cell
            random_cell = xp.random.randint(0, len(ic_locs))
            ic_loc = ic_locs[random_cell]
            ic_angle = ic_angles[random_cell]
            signal_subgrid = signal_grid[int(ic_loc[0]) - grid_pad - 1:int(ic_loc[0]) + grid_pad,
                             int(ic_loc[1]) - grid_pad - 1:int(ic_loc[1]) + grid_pad]
            new_x, new_y, new_angle = jump_immune_cell(signal_subgrid, ic_loc, μ_I, random_numbers_immune_jump[i], L,
                                                       ic_angle=ic_angle, max_std_angle=max_std_angle)
            new_x_tmp = new_x
            new_y_tmp = new_y

            # periodic boundary conditions
            new_x = new_x % (L - 2 * grid_pad)
            new_y = new_y % (L - 2 * grid_pad)

            # old
            if new_x >= L - grid_pad:
                new_x = grid_pad + 1 + xp.mod(new_x, L - grid_pad)
            if new_y >= L - grid_pad:
                new_y = grid_pad + 1 + xp.mod(new_y, L - grid_pad)
            if new_x <= grid_pad:
                new_x = L - grid_pad - 1 - grid_pad + new_x
            if new_y <= grid_pad:
                new_y = L - grid_pad - 1 - grid_pad + new_y

            # update ic_locs
            ic_locs[random_cell] = xp.array([int(new_y), int(new_x)])
            ic_angles[random_cell] = new_angle

            # kill virus where it overlaps with immune cells
            virus_grid[int(new_y), int(new_x)] = False

        # output if called for
        if t_counter < len(t_eval):
            if i == t_eval[t_counter]:
                virus_out[t_counter] = virus_grid.get()
                ic_locs_out[t_counter] = ic_locs.get()
                t_counter += 1

    plt.figure()
    virus_gradient_y, virus_gradient_x = xp.gradient(signal_grid)
    virus_gradient_magnitude = xp.sqrt((virus_gradient_y ** 2 + virus_gradient_x ** 2))
    plt.imshow(virus_gradient_magnitude.get())
    return virus_out, ic_locs_out


def laplacian_kernel():
    return xp.array([[0.25, 0.5, 0.25], [0.5, -3, 0.5], [0.25, 0.5, 0.25]])


def sensing_function(v, grad_v, K):
    if v == 0:
        return 1
    else:
        return K / (K + grad_v / v)