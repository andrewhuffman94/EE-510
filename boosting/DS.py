import numpy as np

class DecisionStump(object):

    def __init__(self):
        self.jmin = None
        self.tht = None
        self.b = None
        self.Fmin = np.inf

    def fit(self, X, y, D):
               
        [m, d] = X.shape
        jmin = self.jmin
        tht = self.tht
        b = self.b
        Fmin = self.Fmin
        
        # b = 1 case: predict 1 if x > theta
        for jj in range(d):
            order = np.argsort(X[:,jj])
            Xj = X[order,jj]
            yj = y[order]
            Dj = D[order]
            F = sum(Dj[y==1])
            if F < Fmin:
                Fmin = F
                jmin = jj
                tht = Xj[0] - 1
            for ii in range(m-1):
                F = F - yj[ii]*Dj[ii]
                if F < Fmin and Xj[ii] != Xj[ii+1]:
                    Fmin = F
                    jmin = jj
                    tht = (Xj[ii] + Xj[ii+1]) / 2
            F = F - yj[-1]*Dj[-1]
            if F < Fmin:
                Fmin = F
                jmin = jj
                tht = Xj[-1] + 1
        jminTop = jmin
        thtTop = tht
        FminTop = Fmin
        
        Fmin = np.inf
        # b = -1 case: predict 1 if x > theta
        for jj in range(d):
            order = np.argsort(X[:,jj])
            Xj = X[order,jj]
            yj = y[order]
            Dj = D[order]
            F = sum(Dj[y==-1])
            if F < Fmin:
                Fmin = F
                jmin = jj
                tht = Xj[0] - 1
            for ii in range(m-1):
                F = F + yj[ii]*Dj[ii]
                if F < Fmin and Xj[ii] != Xj[ii+1]:
                    Fmin = F
                    jmin = jj
                    tht = (Xj[ii] + Xj[ii+1]) / 2
            F = F + yj[-1]*Dj[-1]
            if F < Fmin:
                Fmin = F
                jmin = jj
                tht = Xj[-1] + 1
        jminBot = jmin
        thtBot = tht
        FminBot = Fmin

        if FminTop < FminBot:
            jmin = jminTop
            tht = thtTop
            b = 1
            Fmin = FminTop
        else:
            jmin = jminBot
            tht = thtBot
            b = -1
            Fmin = FminBot

        self.jmin = jmin
        self.tht = tht
        self.b = b
        self.Fmin = Fmin

    def predict(self, X):

        y_pred = np.sign(self.tht - X[:,self.jmin]) * self.b
        return y_pred

    def error(self, y, y_pred, D=None):

        m = len(y)
        if D is None:
            D = np.ones(m)/m
        err = sum(D*(y != y_pred))
        return err
