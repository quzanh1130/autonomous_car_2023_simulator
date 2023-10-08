import numpy as np
import time
import skfuzzy as sf

last_error = 0
Sum_ITerm = 0

class Fuzzy_PID:

    def __init__(self, Pmax, Pmin, Imax, Imin, Dmax, Dmin):
        self.Kpmax = Pmax
        self.Kpmin = Pmin
        self.Kimax = Imax
        self.Kimin = Imin
        self.Kdmax = Dmax
        self.Kdmin = Dmin
        self.sample_time = 0.0
        self.current_time = time.time()
        self.last_time = self.current_time
        self.tfm = self.tfm_generator(-100, 100)
        self.dtfm = self.tfm_generator(-50, 50)
        self.re = self.rule()
        self.rde = self.re.T
        self.rie = self.rule_ki()
        self.a = self.rule_alpha()
        self.b = self.a.T
        self.clear()

    def tfm_generator(self, xmin, xmax):
        x = (xmax - xmin) / 2

        NB = np.array([xmin, xmin, xmin + x / 3], dtype=np.float64)
        NM = np.array([xmin, xmin + x / 3, xmin + x * 2 / 3], dtype=np.float64)
        NS = np.array([xmin + x / 3, xmin + x * 2 / 3, xmin + x], dtype=np.float64)
        ZE = np.array([xmin + x * 2 / 3, xmin + x, xmax - x * 2 / 3], dtype=np.float64)
        PS = np.array([xmin + x, xmax - x * 2 / 3, xmax - x / 3], dtype=np.float64)
        PM = np.array([xmax - x * 2 / 3, xmax - x / 3, xmax], dtype=np.float64)
        PB = np.array([xmax - x / 3, xmax, xmax], dtype=np.float64)

        return [NB, NM, NS, ZE, PS, PM, PB]

    def membership(self, x, tfm):
        x = np.array([x])
        return [sf.trimf(x, tfm[0]), sf.trimf(x, tfm[1]), sf.trimf(x, tfm[2]), \
                sf.trimf(x, tfm[3]), sf.trimf(x, tfm[4]), sf.trimf(x, tfm[5]), sf.trimf(x, tfm[6])]

    def rule(self):
        return np.matrix([[2, 1, 1, 0, 5, 6, 6], [2, 1, 1, 0, 5, 6, 6], [2, 1, 1, 0, 5, 6, 6], \
                          [2, 1, 1, 0, 5, 6, 6], [2, 1, 1, 0, 5, 6, 6], [2, 1, 1, 0, 5, 6, 6], [2, 1, 1, 0, 5, 6, 6]])

    def rule_alpha(self):
        return np.matrix([[2, 2, 2, 2, 2, 2, 2], [3, 3, 2, 2, 2, 3, 3], [4, 3, 3, 2, 3, 3, 4], \
                          [5, 4, 3, 3, 3, 4, 5], [4, 3, 3, 2, 3, 3, 4], [3, 3, 2, 2, 2, 3, 3], [2, 2, 2, 2, 2, 2, 2]])

    def rule_ki(self):
        return np.matrix([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 2, 2, 2, 0, 0], \
                          [0, 2, 4, 2, 4, 2, 0], [0, 0, 2, 2, 2, 0, 0], [0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])

    def clear(self):
        self.SetPoint = 0.0
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0
        self.int_error = 0.0
        self.windup_guard = 10.0
        self.output = 0.0

    def update_K(self, error, d_error):

        # error = 10
        # d_error = 0
        # print(error)
        # print(d_error)

        # print(np.argmax(self.membership(error,self.tfm)))
        # print(np.argmax(self.membership(d_error, self.dtfm)))
        # print(self.re[np.argmax(self.membership(error,self.tfm)),\
        #     np.argmax(self.membership(d_error, self.dtfm))])
        # print(self.re)
        # print(self.re[np.argmax(self.membership(d_error, self.dtfm)),\
        #                np.argmax(self.membership(error,self.tfm))])

        self.Kp = float(self.re[np.argmax(self.membership(d_error, self.dtfm)), \
                                np.argmax(self.membership(error, self.tfm))]) / 6 * (
                              self.Kpmax - self.Kpmin) + self.Kpmin
        self.Kd = self.rde[np.argmax(self.membership(error, self.tfm)),\
             np.argmax(self.membership(d_error, self.dtfm))]/6 *(self.Kdmax-self.Kdmin)+self.Kdmin
        self.alpha = self.a[np.argmax(self.membership(error, self.tfm)),\
             np.argmax(self.membership(d_error, self.dtfm))]
        self.Ki = self.rie[np.argmax(self.membership(error, self.tfm)),\
             np.argmax(self.membership(d_error, self.dtfm))]/4 *(self.Kimax - self.Kimin)+self.Kimin

    def update(self, feedback_value):

        global last_error
        global Sum_ITerm

        # print(last_error)

        error = feedback_value - self.SetPoint
        # print(error)
        delta_time = self.sample_time
        delta_error = error - last_error
        # print(delta_error)

        d_error = delta_error

        self.update_K(error, d_error)
        
        # print("Kp value:"+str(self.Kp))
        # print("Ki value:"+str(self.Ki))
        # print("Kd value:"+str(self.Kd))
        
        #self.Kp = 1
        #self.Ki = 0.01
        #self.Kd = 0.2    ## nho > kp = 3.5 kd = 0.2 ki = 0.01 chay qua duoc bien bao re trái 
        # print("parameter KP: " + str(self.Kp))

        self.PTerm = self.Kp * error

        if self.PTerm > 90:
            self.PTerm = 90
        elif self.PTerm < -90:
            self.PTerm = -90

        #print("parameter self.PTerm: " + str(self.PTerm))
        #print("parameter error: " + str(delta_error))

        self.ITerm = Sum_ITerm + (self.Ki * error * delta_time)
        Sum_ITerm = self.ITerm

        if self.ITerm > 90:
            self.ITerm = 90
        elif self.ITerm < -90:
            self.ITerm = -90

        self.DTerm = self.Kd * delta_error / delta_time

        if self.DTerm > 90:
            self.DTerm = 90
        elif self.DTerm < -90:
            self.DTerm = -90

        last_error = error

        Output = self.PTerm + (self.ITerm) + (self.DTerm)
        #print("parameter Output: " + str(Output))
        #self.output = Output

        if Output > 90:
            self.output = 90
        elif Output < -90:
            self.output = -90
        else:
            self.output = Output

    def setKp(self, Pmax, Pmin):
        self.Kpmax = Pmax
        self.Kpmin = Pmin

    def setKd(self, Dmax, Dmin):
        self.Kdmax = Dmax
        self.Kdmin = Dmin

    def setKi(self, Imax, Imin):
        self.Kimax = Imax
        self.Kimin = Imin

    def setSampleTime(self, sample_time):
        self.sample_time = sample_time

    def setSetPoint(self, setpoint):
        self.SetPoint = setpoint

class PID:
    def __init__(self, P = 0.2, I = 0, D = 0):
        '''
        Initialization.
        :param P: Proportional Parameter
        :param I: integral Parameter
        :param D: Derivative Parameter
        '''
        self.Kp, self.Ki, self.Kd = P, I, D
        self.sample_time = 0.0
        self.current_time = time.time()
        self.last_time = self.current_time
        self.clear()
    
    def clear(self):
        '''
        Clear all parameters.
        '''
        self.SetPoint = 0.0
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        self.int_error = 0.0
        # self.windup_guard = 15.0
        self.windup_guard = 100.0

        self.output = 0.0

    def update(self, feedback_value):
        '''
        State Update.
        :param feedback_value: Current state value
        '''
        global last_error
        global Sum_ITerm

        # print(last_error)

        error =  feedback_value - self.SetPoint

        # print(error)

        delta_time  = self.sample_time
        
    
        delta_error = error - last_error
        # print(delta_error)

       
        self.PTerm = self.Kp * error

        if self.PTerm > 90:
            self.PTerm = 90
        elif self.PTerm < -90:
            self.PTerm = -90

        self.ITerm = Sum_ITerm + (self.Ki * error * delta_time)
        Sum_ITerm = self.ITerm 

        self.DTerm = self.Kd * delta_error / delta_time

        if self.DTerm > 90:
            self.DTerm = 90
        elif self.DTerm < -90:
            self.DTerm = -90
        
        last_error = error

        Output = self.PTerm + (self.ITerm) + (self.DTerm)
        
        if Output > 90:
            self.output = 90
        elif Output < -90:
            self.output = -90
        else:
            self.output = Output

    def setKp(self, Proportional_gain):
        self.Kp = Proportional_gain
    
    def setKi(self, Integral_gain):
        self.Ki = Integral_gain

    def setKd(self, derivative_gain):
        self.Kd = derivative_gain
    
    def setSampleTime(self, sample_time):
        self.sample_time = sample_time
    
    def setSetPoint(self, setpoint):
        self.SetPoint = setpoint