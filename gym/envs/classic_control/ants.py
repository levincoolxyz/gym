import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class AntsEnv(gym.Env):
    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.

    The main API methods that users of this class need to know are:

        step
        reset
        render
        close
        seed

    And set the following attributes:

        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards

    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.

    The methods are accessed publicly as "step", "reset", etc.. The
    non-underscored versions are wrapper methods to which we may add
    functionality over time.
    """

    # Set this in SOME subclasses
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    reward_range = (-float('inf'), float('inf'))
    spec = None

    def __init__(self, Nmax=12, dt=0.1, 
        f0=0.25, b=1., gamma=1., grot=1., fkin0=1., tkin0=1., beta=2., dphi=30., goalDir=0):
        self.Nmax = Nmax # total number of ants (including one informer)
        self.dt = dt # decision time step

        self.f0 = f0 # ant pull force
        self.b = b # radius of cargo
        self.gamma = gamma # translational response coeff of cargo
        self.grot = grot # rotational response coeff of cargo
        self.fkin0 = fkin0 # bare translational friction force
        self.tkin0 = tkin0 # bare rotational friction force
        self.beta = beta/self.Nmax # lifter friction reduction coeff
        self.dphi = dphi/180*np.pi # max pull angle change
        self.goalDir = goalDir/180*np.pi; # informer target direction

        self.theta = np.linspace(0.,2*np.pi,self.Nmax+1)
        self.theta = self.theta[:-1] # ant angular locations
        self.rx = self.b*np.cos(self.theta) # initial ant locations
        self.ry = self.b*np.sin(self.theta) # initial ant locations
        self.phi = np.full(self.Nmax,0.) # initial ant directions
        self.pulling = np.full(self.Nmax,0) # initially ants not pulling in graphics

        """
        self.action_space = spaces.Dict({
            "liftProb": spaces.Box(low=0.,high=1., shape=(self.Nmax-1,), dtype=np.float32), 
            "phi": spaces.Box(low=-self.dphi/2, high=self.dphi/2, shape=(self.Nmax-1,), dtype=np.float32)
            })

        self.action_space = spaces.Dict({
            "isPuller": spaces.Discrete(self.Nmax-1), 
            "phi": spaces.Box(low=-self.dphi/2, high=self.dphi/2, shape=(self.Nmax-1,), dtype=np.float32)
            })
        """

        # combine probability to pull with pulling angle adjustment phi
        self.action_space = spaces.Box(low=-1., high=1., shape=((self.Nmax-1)*2,), dtype=np.float32)

        # dot product between ant direction and force direction due to other ants
        high = np.tile([np.inf,1],self.Nmax-1)
        low = np.tile([0,-1],self.Nmax-1)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.state = None, None

        self.viewer = None

        self.seed()

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        position, velocity = self.state

        # untangle action space order
        acts = list(range((self.Nmax-1)*2))
        inverse = np.append(acts[:-1:2],acts[1::2])
        action = action[inverse]

        # liftProb = np.append(0, action["liftProb"])
        liftProb = np.append(0, action[:self.Nmax-1]/2.+.5)
        isPuller = np.random.uniform(0., 1., self.Nmax)
        isPuller = (isPuller >= liftProb)*1

        self.pulling = np.logical_xor(isPuller, self.pulling)

        # self.phi = np.append(-position[2]-self.theta[0]+self.goalDir, action["phi"])
        self.phi = np.append(-position[2]-self.theta[0]+self.goalDir, action[self.Nmax-1:]*self.dphi/2.)

        # update ant states
        self.rx = self.b*np.cos(position[2] + self.theta)
        self.ry = self.b*np.sin(position[2] + self.theta)
        ax = np.cos(position[2] + self.theta + self.phi)
        ay = np.sin(position[2] + self.theta + self.phi)
        pulx = isPuller*ax
        puly = isPuller*ay

        # calculate relative forces
        fkin = np.max([self.fkin0*(1. - self.beta*np.sum(1 - isPuller)), 0.])
        tkin = np.max([self.tkin0*(1. - self.beta*np.sum(1 - isPuller)), 0.])
        fcmx = (self.f0 - fkin)*np.sum(pulx)
        fcmy = (self.f0 - fkin)*np.sum(puly)
        tcm = (self.f0 - tkin)*np.sum(isPuller*np.sin(self.phi))
        velx = fcmx/self.gamma
        vely = fcmy/self.gamma
        omega = tcm/self.grot
        frotx = self.grot/self.b*self.ry*omega
        froty = -self.grot/self.b*self.rx*omega
        Ftotx = fcmx - frotx
        Ftoty = fcmy - froty

        # update cargo state
        velocity = np.array([velx, vely, omega])
        position = position + velocity*self.dt
        self.state = position, velocity
        self.Ftot = np.linalg.norm([Ftotx, Ftoty], axis=0)
        Fnorm = self.Ftot
        Fnorm[Fnorm==0] = 1
        # print(np.mean(self.Ftot))
        Fcos = np.sum([Ftotx*ax/Fnorm, Ftoty*ay/Fnorm], axis=0)
        Fsin = np.sum([Ftotx*ay/Fnorm, -Ftoty*ax/Fnorm], axis=0)
        self.Fang = np.arctan2(Fsin,Fcos)/np.pi
        # print([np.min(self.Fang),np.max(self.Fang)])

        reward = velocity[0]*np.cos(self.goalDir) + velocity[1]*np.sin(self.goalDir)

        return self._get_obs(), reward, False, {}

    def reset(self, rand=False):
        """Resets the state of the environment and returns an initial observation.

        Returns:
            observation (object): the initial observation.
        """

        self.state = np.full(3, 0.), np.full(3, 0.) # reset cargo location
        self.pulling = np.full(self.Nmax,0) # initialize ants rendering state

        if rand:
            self.goalDir = np.random.uniform(-np.pi,np.pi); # informer direction
            antSpace = 2*np.pi/self.Nmax
            antWidth = antSpace/2
            for i in range(self.Nmax): # ant distribution around cargo
                self.theta[i] = np.random.uniform(0.,antSpace-antWidth/2) + antSpace*i + antWidth/2
            np.random.shuffle(self.theta)
            self.phi = np.random.uniform(self.dphi/2-self.dphi/2.,2*np.pi,(self.Nmax,)) # initial ant directions
        else:
            self.theta = np.linspace(0.,2*np.pi,self.Nmax+1)
            self.theta = self.theta[:-1] # ant angular locations
            self.phi = np.full(self.Nmax,0.) # initial ant directions

        self.rx = self.b*np.cos(self.theta) # initial ant locations
        self.ry = self.b*np.sin(self.theta) # initial ant locations

        self.Ftot = np.full(self.Nmax, 0.)
        self.Fang = np.full(self.Nmax, 0.)
        return self._get_obs()

    def _get_obs(self):
        ants = np.array(range(self.Nmax-1))
        order = np.concatenate((ants[:,None],ants[:,None]+self.Nmax-1),axis=1).flatten()
        obs = np.append(self.Ftot[1:], self.Fang[1:])
        return obs[order]

    def render(self, mode='human'):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
        """

        from gym.envs.classic_control import rendering
        from pyglet.gl import glRotatef, glPushMatrix

        class Flip(rendering.Transform):
            def __init__(self, flipx=False, flipy=False):
                self.flipx = flipx
                self.flipy = flipy
            def enable(self):
                glPushMatrix()
                if self.flipx: glRotatef(180, 0, 1., 0)
                if self.flipy: glRotatef(180, 1., 0, 0)

        def make_ellipse(major=10, minor=5, res=30, filled=True):
            points = []
            for i in range(res):
                ang = 2*np.pi*i / res
                points.append((np.cos(ang)*major, np.sin(ang)*minor))
            if filled:
                return rendering.FilledPolygon(points)
            else:
                return rendering.PolyLine(points, True)

        def make_ant(length=1):
            l = length/4
            leg1 = rendering.PolyLine([(l/2,0),(l/2,l),(-l*.5,3*l),(l/2,l)],True)
            leg2 = rendering.PolyLine([(l/2,0),(l/2,-l),(-l*1.5,-3*l),(l/2,-l)],True)
            leg3 = rendering.PolyLine([(l*1.5,0),(l*1.5,l/2),(l,2.5*l),(l*1.5,l/2)],True)
            leg4 = rendering.PolyLine([(l*1.5,0),(l*1.5,-l/2),(l*2,-2.5*l),(l*1.5,-l/2)],True)
            leg5 = rendering.PolyLine([(l*2.5,0),(l*2.5,l/2),(l*2.5,2*l),(l*2.5,l/2)],True)
            leg6 = rendering.PolyLine([(l*2.5,0),(l*2.5,-l/2),(l*3.5,-2*l),(l*2.5,-l/2)],True)
            circ0 = make_ellipse(2*l, l/1.5)
            circ1 = make_ellipse(l, l/2)
            circ1.add_attr(rendering.Transform(translation=(l+l/2, 0)))
            circ2 = make_ellipse(l, l)
            circ2.add_attr(rendering.Transform(translation=(2*l+l/2, 0)))
            geom = rendering.Compound([leg1, leg2, leg3, leg4, leg5, leg6, circ0, circ1, circ2])
            geom.add_attr(rendering.Transform(translation=(-3*l, 0)))
            geom.add_attr(Flip(flipx=True))
            return geom

        def draw_ant(Viewer, length=1, **attrs):
            geom = make_ant(length=length)
            rendering._add_attrs(geom, attrs)
            Viewer.add_onetime(geom)
            return geom

        def draw_lasting_line(Viewer, start, end, **attrs):
            geom = rendering.Line(start, end)
            rendering._add_attrs(geom, attrs)
            Viewer.add_geom(geom)
            return geom

        def draw_lasting_circle(Viewer, radius=10, res=30, filled=True, **attrs):
            geom = rendering.make_circle(radius=radius, res=res, filled=filled)
            rendering._add_attrs(geom, attrs)
            Viewer.add_geom(geom)
            return geom

        def draw_filled_donut(Viewer, rout=10, rin=5, res=30, **attrs):
            points = []
            for i in range(res+1):
                ang = 2*np.pi*i / res
                points.append((np.cos(ang)*rout, np.sin(ang)*rout))
            for i in range(res+1):
                ang = 2*np.pi*i / res
                points.append((np.cos(ang)*rin, np.sin(ang)*rin))
            geom = []
            for i in range(res):
                geom.append(rendering.FilledPolygon([points[j] for j in [i,i+1,i+res+2,i+res+1]]))
            geom = rendering.Compound(geom)
            rendering._add_attrs(geom, attrs)
            Viewer.add_onetime(geom)
            return geom

        if self.viewer is None:
            self.viewer = rendering.Viewer(800,400)
            draw_lasting_line(self.viewer, (-1000., 0), (1000., 0))
            draw_lasting_line(self.viewer, (0, -1000.), (0, 1000.))
            draw_lasting_line(self.viewer, (-800., -800.), (800., 800.))
            draw_lasting_line(self.viewer, (800., -800.), (-800., 800.))
            startPoint = draw_lasting_circle(self.viewer, radius=1)
            startPoint.set_color(.8, .8, .8)

        position, velocity = self.state
        bound = 10
        self.viewer.set_bounds(-bound+position[0], bound+position[0], 
            -bound/2+position[1], bound/2+position[1])

        if position is None: return None

        trail = draw_lasting_line(self.viewer, (position[0], position[1]), 
            (position[0]+velocity[0]*self.dt, position[1]+velocity[1]*self.dt))
        trail.set_color(.9, .1, .1)

        cargo = draw_filled_donut(self.viewer, rout=self.b, rin=self.b/2.5)
        cargo.add_attr(rendering.Transform(rotation=position[2], 
            translation=(position[0],position[1])))
        cargo.set_color(.8, .4, .2)

        leader = self.viewer.draw_line((0., 0.), (self.b, 0.))
        leader.add_attr(rendering.Transform(rotation=position[2]+self.theta[0], 
            translation=(position[0],position[1])))

        for i in range(self.Nmax):
            ant = draw_ant(self.viewer, self.b/self.Nmax*np.pi)
            ant.add_attr(Flip(flipy=self.pulling[i]))
            antMove = rendering.Transform(rotation=position[2]+self.theta[i]+self.phi[i], 
                translation=(self.rx[i] + position[0],self.ry[i] + position[1]))
            ant.add_attr(antMove)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """

        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)