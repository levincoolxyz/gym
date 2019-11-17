import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from scipy import integrate

class SeastarEnv(gym.Env):

    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    Fmax_push = 2
    Fmax_pull = 2
    kp = 1
    lo = 1.5
    lmin = 1
    lmax = 2
    lc_pull = 1.1
    lc_push = 1.9

    def __init__(self, Nfeet = 2, dt = 0.1, Lbody = 0, SensorMode = "ltxy"):
        self.Nfeet = Nfeet
        self.dt = dt
        self.Lbody = Lbody
        self.mg = 0.9
        self.t_recovery = 0
        self.cx = 1
        self.cy = 1
        self.cb = 10
        gamma = 10
        self.M = self.mg/gamma
        self.J = (1/4)*self.M*(self.Lbody/2)**2
        self.d = np.linspace(-self.Lbody/2,self.Lbody/2, num = self.Nfeet, endpoint = False) + self.Lbody/self.Nfeet/2

        # Initial Conditions for cg
        self.rx = 0
        self.ry = 1
        self.beta = 0
        
        # choose sensory capabilities
        self.sensor = SensorMode
        if self.sensor.casefold() == "ltxy".casefold():
            high = np.repeat(np.array([4,np.pi/2,np.finfo(np.float32).max,np.finfo(np.float32).max]),self.Nfeet)
            low = np.repeat(np.array([0,-np.pi/2,-np.finfo(np.float32).max,np.finfo(np.float32).max]),self.Nfeet)
        elif self.sensor.casefold() == "lty".casefold():
            high = np.repeat(np.array([4,np.pi/2,np.finfo(np.float32).max]),self.Nfeet)
            low = np.repeat(np.array([0,-np.pi/2,-np.finfo(np.float32).max]),self.Nfeet)
        elif self.sensor.casefold() == "lt".casefold():
            high = np.repeat(np.array([4,np.pi/2]),self.Nfeet)
            low = np.repeat(np.array([4,-np.pi/2]),self.Nfeet)
        elif self.sensor.casefold() == "l".casefold():
            high = np.repeat(np.array(4),self.Nfeet)
            low = np.repeat(np.array(0),self.Nfeet)
        else:
            raise NotImplementedError
        
        self.observation_space = spaces.Box(low=-low, high=high, dtype=np.float32)

        self.action_space = spaces.Tuple(tuple(spaces.Discrete(2) for _ in range(self.Nfeet)))

        self.viewer = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        terminal = False
        dt = self.dt
        t_current = 0
        options = {'rtol':1e-4,'atol':1e-8,'max_step': 1e-2}
        body , theta = self.state
        action = np.array(action)>0
        dettach_bool = action
        self.actionNotDone = np.max(action*(~(dettach_bool<0)))
        self.x_att[dettach_bool] = body[0] + self.d[dettach_bool]*np.cos(body[2]) + body[1]*np.tan(np.pi/6)
        sol = integrate.solve_ivp(self._walk, (t_current,dt), body, method='RK45', t_eval=None, dense_output=False, events=None, vectorized=False,**options)
        t_current = sol.t[-1]
        body = sol.y[:,-1]
        
        x = body[0];
        y = body[1];
        beta = body[2];
        
        # body attachment point of each foot
        xb = x + self.d*np.cos(beta)
        yb = y + self.d*np.sin(beta)
        
        
        # compute dx for each foot
        dx = self.x_att - xb
        
        # compute theta for each foot
        theta = np.arctan(dx/yb)
        self.state = body, theta
        terminal = self._terminal(body)
        self.time = self.time + dt
        
        reward = body[3]*np.exp(np.abs(body[0]))*np.exp(-(body[1]-1)**2/2/(.5**2))

        return self._get_obs(), reward, terminal, {}

    def _walk(self,t,X):
        
        x = X[0];
        y = X[1];
        beta = X[2];
        
        # body attachment point of each foot
        xb = x + self.d*np.cos(beta)
        yb = y + self.d*np.sin(beta)
        
        # compute dx for each foot
        dx = self.x_att - xb
      
        # compute length of each foot 
        self.l = np.sqrt(np.square(yb) + np.square(dx))
        
        # compute theta for each foot
        theta = np.arctan(dx/yb)
               
        # compute the forces in each foot
        Fa = self._F_active(theta)
        Fp = self._F_passive()
        Fn = self._F_nonlinear()
        F = Fa + Fp + Fn

        Fx = -np.sum(F*np.sin(theta))
        Fy = np.sum(F*np.cos(theta))
        Fb = np.sum(F*np.cos(theta)*self.d*np.cos(beta) + F*np.sin(theta)*self.d*np.sin(beta))

        dX = X[3]
        dY = X[4]
        dBeta = X[5]
        
        d2X = (Fx - self.cx*dX)*(1/self.M)
        d2Y = (Fy - self.cy*dY - self.mg)*(1/self.M)
        d2Beta = 0
        if (self.J>0):
            d2Beta = (Fb - self.cb*dBeta)*(1/self.J)

        return np.array([dX, dY, dBeta, d2X, d2Y, d2Beta])     
    
    def _terminal(self,body):
        return (body[1]<0) or (np.max(self.l)>3) or (body[0]<-3) or (np.linalg.norm(body[3:4])<5e-4)

    def reset(self, rand=True):

        # initial leg configurations
        if rand:
            self.theta = np.linspace(-1.,1., self.Nfeet)*np.pi/5 + np.random.uniform(np.full(self.Nfeet,-1),np.full(self.Nfeet,1))*np.pi/20
        else:
            self.theta = np.linspace(-1.,1., self.Nfeet)*np.pi/6

        self.x_att = self.rx + self.d + self.ry*np.tan(self.theta)
        xb = self.rx + self.d*np.cos(self.beta)
        yb = self.ry + self.d*np.sin(self.beta)

        # compute dx for each foot
        dx = self.x_att - xb

        # compute length of each foot 
        self.l = np.sqrt(np.square(yb) + np.square(dx))
        self.state = np.array([self.rx,self.ry,self.beta,0,0,0]), self.theta
        self.time = 0
        self.actionNotDone = False
        return self._get_obs()

    def _get_obs(self):
        if self.sensor.casefold() == "l".casefold():
            return self.l
        elif self.sensor.casefold() == "lt".casefold():
            body, theta = self.state
            return np.stack([self.l,theta]).transpose().reshape(-1,)
        elif self.sensor.casefold() == "lty".casefold():
            body, theta = self.state
            feetYVel = body[4] + self.d*body[5]*np.cos(body[2])
            return np.stack([self.l,theta,feetYVel]).transpose().reshape(-1,)
        elif self.sensor.casefold() == "ltxy".casefold():
            body, theta = self.state
            feetXVel = body[3] + self.d*body[5]*np.sin(body[2])
            feetYVel = body[4] + self.d*body[5]*np.cos(body[2])
            return np.stack([self.l,theta,feetXVel,feetYVel]).transpose().reshape(-1,)
        else:
            raise NotImplementedError

    def render(self, mode='human'):
        
        body, theta = self.state
        from gym.envs.classic_control import rendering
        
        if self.viewer is None:
            self.viewer = rendering.Viewer(800,400)
        
        bound = 5*self.lmin
        self.viewer.set_bounds(-bound+body[0],bound+body[0],-bound/2+body[1],bound+body[1])
        
        self.viewer.draw_line((-1000., 0), (1000., 0))
        
        for i in range(1,100):
            l,b,t,r = -.075-i*5, 0, 5, .075-i*5
            post = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            post.set_color(.8, .5, .5)
            
        for i in range(100):
            l,b,t,r = -.075+i*5, 0, 5, .075+i*5
            post = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            post.set_color(.8, .8, .8)
        
        for i in range(self.Nfeet):
            l,b,t,r = 0, -.1, .1, self.l[i]
            foot = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            ftTrans = rendering.Transform(rotation=theta[i]-np.pi/2,translation=(self.d[i]*np.cos(body[2])+body[0],self.d[i]*np.sin(body[2])+body[1]))
            foot.add_attr(ftTrans)
            if i%3 == 0: foot.set_color(.9, .1, .1)
            if i%3 == 1: foot.set_color(.1, .9, .1)
            if i%3 == 2: foot.set_color(.1, .1, .9)
            
        bodyTrans = rendering.Transform(rotation=body[2], translation=(body[0],body[1]))
        if self.Lbody > 0: 
            l,b,t,r = -self.Lbody/2, -.18, .18, self.Lbody/2
            sstar = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            sstar.add_attr(bodyTrans)
            sstar.set_color(.5, .5, .5)

        center = self.viewer.draw_circle(radius=.2)
        center.add_attr(bodyTrans)
        center.set_color(.5, .5, .5)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def _F_active(self,theta):

        #active element
        Fa = np.zeros(self.Nfeet)
        push_bool = (theta < 0)

        # push
        push1 = push_bool*(self.l >= self.lmin)*(self.l < self.lc_push)
        push_fun1 = (self.l-self.lmin)*(self.Fmax_push/(self.lc_push - self.lmin))

        Fa[push1] = push_fun1[push1]
        push2 = push_bool*(self.l >= self.lc_push)*(self.l < self.lmax)
        push_fun2 = (self.lmax - self.l)*(self.Fmax_push/(self.lmax - self.lc_push))

        Fa[push2] = push_fun2[push2]
        pull_bool = (theta > 0)

        # pull
        pull1 = pull_bool*(self.l <= self.lmax)*(self.l > self.lc_pull)
        pull_fun1 = (self.l-self.lmax)*(self.Fmax_pull/(self.lmax - self.lc_pull))

        Fa[pull1] = pull_fun1[pull1]
        pull2 = pull_bool*(self.l <= self.lc_pull)*(self.l > self.lmin)
        pull_fun2 = (self.lmin - self.l)*(-self.Fmax_pull/(self.lmin - self.lc_pull))

        Fa[pull2] = pull_fun2[pull2]

        return Fa

    def _F_passive(self):
        # passive element
        Fp = -self.kp*(self.l-self.lo)
        return Fp

    def _F_nonlinear(self):
        return 0