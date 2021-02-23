'''
A simulation written as a response to https://www.youtube.com/watch?v=YrG5THXQesU&t=660s
Original source: 
2021. feb. 23.
Kürti Zoltán

Feel free to modify and/or redistribute the code, please include this notice if you do so.
'''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as anm
from scipy import integrate

duration = 20 #video length (s)
FPS = 60

M1 = 10   #mass of ship 1
M2 = 15   #mass of ship 2
Mr = 1      #mass of the rope
L = 1.4      #stationary length of the rope
D = 500     #spring constant of the rope
N = 10     #number of point masses, including the two ships and the N-2 points of the rope

strength = 1

Fthrust = 1  #thrust of the rocket
tstart = 1      #engine start time
burnTime = 10  #engine burn length

flameScale = 0.1 #cosmetic parameter

def thruster(r, v, t):
    """
    r: position of the points
    v: velocity of the points
    ret: thrust of the first ship, thrust of the second ship
    """
    
    if t > tstart and t < tstart + burnTime:
        rocketDirection = r[-2:] - r[:2]
        rocketDirection /= np.sqrt(np.sum(rocketDirection**2))
        thrustDirection = np.array([-rocketDirection[1], rocketDirection[0]])
    else:
        thrustDirection = np.array([0, 0])
    return -thrustDirection * Fthrust, thrustDirection * Fthrust

def F(r, v):
    r = np.array(r)
    v = np.array(v)
    '''
    r: numpy array, containing the current position of points
        [x0, y0, x1, y1, ...]
    
    v: numpy array, containing the current velocity of points
        [v_x0, v_y0, v_x1, v_y1, ...]
    
    return: numpy array, containing the force acting on the points
        [F_x0, F_y0, F_x1, F_y1, ...]
    '''
    ret = np.zeros((2*N))
    
    #calculating the spring force between the points:
    
    #displacement[i] = r[i+1] - r[i]
    displacement = r[2::1] - r[0:-2:1]
    relelocity = v[2::1] - v[0:-2:1]
    #temp[i] = sqrt(displacement[2i]^2 + displacement[2i+1]^2)
    temp = np.sqrt(displacement[0::2]**2 + displacement[1::2]**2)
    #dispLength[2i] = dispLength[2i+1] = temp[i]
    dispLength = np.zeros((2*(N-1)))
    dispLength[0::2] = temp
    dispLength[1::2] = temp
    #elongation[i] = displacement[i] - l * (direction of elongation[i])
    elongation = displacement - l * displacement / dispLength
    #Fspring[i] = elongation[i] * d
    Fspring = elongation * d
    #F[i] = Fspring[i] - Fspring[i+1]
    ret += np.append(Fspring, [0, 0]) - np.append([0, 0], Fspring)
    
    #calculatig the dissipative forces due to internal friction of the tether
    
    #calculate the angular velocity of bending at each internal point
    #cos(alpha) = r1*r2 / (|r1|*|r2|)
    #-alpha' * sin(alpha) = (v1*r2 + r1*v2) / (|r1|*|r2|) - (r1*v1 * |r2| + |r1| * r2*v2) * r1*r2 / (|r1|*|r2|)^2
    #alpha' = -((v1*r2 + r1*v2) / (r1*r2) - (r1*v1 * |r2| + |r1| * r2*v2) / (|r1|*|r2|))
    
    '''
    FRICTION CALCULATION NEEDED HERE TO CALCULATE OSCILLATION DAMPENING
    '''
    
    return ret
    
def func(state, t):
    """
    state: array, containing the position and velocity of the points
        [x0, y0, x1, y1, ..., v_x0, v_y0, v_x1, v_y1, ...]
    
    t: time in the simulation
    
    return: the first time derivative of state
    """
    
    r = state[0:2*N]
    v = state[2*N:]
    
    #tether
    force = F(r, v)
    
    #thrusters
    thrust1, thrust2 = thruster(r, v, t)
    force[0:2] += thrust1
    force[-2:] += thrust2
    
    return np.append(v, force / masses)

def animate(i):
    x = result[i][0:2*N:2]
    y = result[i][1:2*N:2]
    r = result[i][0:2*N]
    v = result[i][2*N:]
    t = times[i]
    
    thrust1, thrust2 = thruster(r, v, t)
    flame1.set_data([x[ 0], x[ 0] - flameScale * thrust1[0]],[y[ 0], y[ 0] - flameScale * thrust1[1]])
    flame2.set_data([x[-1], x[-1] - flameScale * thrust2[0]],[y[-1], y[-1] - flameScale * thrust2[1]])
    
    tether.set_data(x, y)
    
    ship1.set_data(x[0], y[0])
    ship2.set_data(x[-1], y[-1])
    
    return [tether, ship1, ship2, flame1, flame2]

def render(name = None):
    global Fthrust
    global burnTime
    global l
    global d
    global masses
    global result
    global times
    global flame1
    global flame2
    global tether
    global ship1
    global ship2
    Fthrust *= strength
    burnTime /= strength

    if N > 2:
        m = Mr / (N-2) #mass of a single point in the rope
    l = L / (N-1)  #stationary length between two points
    d = D * (N-1)  #spring constant between two points of the rope

    #setting up the mass array
    #   [M1, M1, m, m, ..., M2, M2]
    masses = np.zeros((2*N))
    if N > 2:
        masses += m
    masses[0] = masses[1] = M1
    masses[-2] = masses[-1] = M2

    #setting up initial state
    v0 = 0.3
    initState = np.zeros(4*N)
    for i in range(N):
        initState[2*i] = i * l - L/2
        initState[2*i+1] = 0
    
    times = np.linspace(0, duration, int(duration*FPS))
    result = integrate.odeint(func, initState, times)

    fig = plt.figure(figsize = [16, 10], dpi = 108)
    ax = plt.axes(xlim=(-1.6, 1.6), ylim=(-1,1))
    plt.axis("off")
    plt.plot()
    tether, = ax.plot([], [], "b-", lw=3)
    ship1, = ax.plot([], [], "bo", ms=2*M1)
    ship2, = ax.plot([], [], "bo", ms=2*M2)
    flame1, = ax.plot([], [], "r-", lw=6)
    flame2, = ax.plot([], [], "r-", lw=6)
    
    anim = anm.FuncAnimation(fig, animate, duration*FPS, interval = 1000/FPS)
    if name != None:
        anim.save(name)
    else:
        plt.show()

strength = 1
render()
strength = 10
render()